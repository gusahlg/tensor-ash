//! Static SASS census: opcode classes, memory-op widths, loops, and the
//! Volta/Turing/Ampere/Ada 128-bit control bits.
//!
//! Input is `nvdisasm -b SMxx <ucode.bin>` text (for mnemonics) plus the
//! raw UCODE bytes (for the control bits, which nvdisasm does not
//! print).  Control-bit layout (sm_70 .. sm_89), bits of the 128-bit
//! little-endian instruction word:
//!
//! ```text
//!   [104:0]    opcode + operands
//!   [108:105]  stall cycles (0-15) before this warp may issue again
//!   [109]      yield bit (0 = yield hint, CuAssembler prints 'Y')
//!   [112:110]  write scoreboard set by this instr (7 = none)
//!   [115:113]  read scoreboard set by this instr (7 = none)
//!   [121:116]  wait mask: scoreboards to wait on before issue
//!   [125:122]  operand reuse-cache flags
//! ```
//!
//! Sources: CuAssembler `CuControlCode.splitCode` / `CuSMVersion`
//! (`CCPos_7x_8x = 64 + 41`), maxas control-code wiki, Jia et al.,
//! "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
//! (arXiv:1804.06826).  The decode is cross-checked in [`Census::sanity`]:
//! variable-latency ops (LDG/LDS/LDSM/S2R/...) must set a write
//! scoreboard, fixed-latency ALU ops must not, and the NOP padding
//! after EXIT must decode to CuAssembler's `B------:R-:W-:Y:S00`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::{Context, Result, bail};

/// Decoded control bits of one instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ctrl {
    pub(crate) stall: u8,
    /// Raw yield bit; `0` means the yield hint is set.
    pub(crate) yield_bit: u8,
    pub(crate) wbar: u8,
    pub(crate) rbar: u8,
    pub(crate) wait: u8,
    pub(crate) reuse: u8,
}

impl Ctrl {
    pub(crate) fn decode(word: &[u8]) -> Self {
        let hi = u64::from_le_bytes(word[8..16].try_into().expect("16-byte instruction"));
        let c = hi >> 41; // bit 105 of the 128-bit word
        Self {
            stall: (c & 0xf) as u8,
            yield_bit: ((c >> 4) & 1) as u8,
            wbar: ((c >> 5) & 7) as u8,
            rbar: ((c >> 8) & 7) as u8,
            wait: ((c >> 11) & 0x3f) as u8,
            reuse: ((c >> 17) & 0xf) as u8,
        }
    }

    /// CuAssembler-style string, e.g. `B0----5:R-:W1:Y:S04`.
    pub(crate) fn render(&self) -> String {
        let wait: String = (0..6)
            .map(|i| {
                if self.wait & (1 << i) != 0 {
                    char::from(b'0' + i)
                } else {
                    '-'
                }
            })
            .collect();
        let sb = |v: u8| {
            if v == 7 {
                "-".to_string()
            } else {
                v.to_string()
            }
        };
        format!(
            "B{wait}:R{}:W{}:{}:S{:02}",
            sb(self.rbar),
            sb(self.wbar),
            if self.yield_bit == 0 { 'Y' } else { '-' },
            self.stall
        )
    }
}

/// One disassembled instruction.
#[derive(Clone, Debug)]
pub(crate) struct Instr {
    pub(crate) addr: u32,
    pub(crate) pred: Option<String>,
    /// Full mnemonic with modifiers, e.g. `LDG.E.128.CONSTANT`.
    pub(crate) op: String,
    pub(crate) operands: String,
    pub(crate) ctrl: Ctrl,
}

impl Instr {
    /// Base opcode before the first '.', e.g. `LDG`.
    pub(crate) fn base(&self) -> &str {
        self.op.split('.').next().unwrap_or(&self.op)
    }

    fn has_mod(&self, m: &str) -> bool {
        self.op.split('.').skip(1).any(|x| x == m)
    }

    /// Access width in bits for load/store opcodes.
    pub(crate) fn mem_width(&self) -> u32 {
        if self.base() == "LDSM" {
            // .M88.N: N 8x8 b16 matrices per warp = 16*N bytes/thread... per
            // thread N 32-bit registers.
            return match self.op.rsplit('.').next() {
                Some("4") => 128,
                Some("2") => 64,
                _ => 32,
            };
        }
        for (m, w) in [
            ("128", 128),
            ("64", 64),
            ("U16", 16),
            ("S16", 16),
            ("U8", 8),
            ("S8", 8),
        ] {
            if self.has_mod(m) {
                return w;
            }
        }
        32
    }

    /// Multiply-adds per warp for one tensor-core MMA (`HMMA.16816` =
    /// m16 n8 k16 = 2048), 0 for anything else.
    pub(crate) fn mma_fmas(&self) -> u64 {
        if !matches!(self.base(), "HMMA" | "IMMA") {
            return 0;
        }
        match self.op.split('.').nth(1) {
            Some("16816") => 16 * 8 * 16,
            Some("1688") => 16 * 8 * 8,
            Some("16832") => 16 * 8 * 32,
            Some("884") => 8 * 8 * 4,
            _ => 0,
        }
    }

    /// Absolute branch target for `BRA 0x...` (raw-mode nvdisasm).
    pub(crate) fn branch_target(&self) -> Option<u32> {
        if self.base() != "BRA" {
            return None;
        }
        let tok = self
            .operands
            .split([' ', ','])
            .find(|t| t.starts_with("0x"))?;
        u32::from_str_radix(tok.trim_start_matches("0x"), 16).ok()
    }
}

/// Opcode classes for the census.
pub(crate) fn class_of(base: &str) -> &'static str {
    match base {
        "FFMA" => "ffma",
        "FMUL" | "FADD" | "FMNMX" | "FSETP" | "FSEL" | "FSET" | "FCHK" | "FRND" => "fp32-other",
        "HMMA" | "IMMA" | "DMMA" | "BMMA" => "mma",
        "HFMA2" | "HADD2" | "HMUL2" | "HMNMX2" | "HSETP2" | "HSET2" => "fp16x2",
        "F2F" | "F2FP" | "I2F" | "F2I" | "I2I" | "I2IP" | "FRND.F16" => "convert",
        "MUFU" => "mufu",
        "IMAD" | "IADD3" | "LEA" | "ISETP" | "SEL" | "LOP3" | "SHF" | "IABS" | "IMNMX" | "PRMT"
        | "FLO" | "POPC" | "BREV" | "IDP" | "IMUL" | "LOP" | "IADD" | "SGXT" | "BMSK" => "int",
        "MOV" | "P2R" | "R2P" | "PLOP3" | "R2UR" | "SHFL" | "VOTE" | "VOTEU" | "MATCH" => {
            "move/pred"
        }
        "LDG" => "ldg",
        "STG" => "stg",
        "LDS" => "lds",
        "STS" => "sts",
        "LDSM" => "ldsm",
        "LDL" => "ldl(spill)",
        "STL" => "stl(spill)",
        "LDC" | "LD" | "ST" | "ATOM" | "ATOMG" | "ATOMS" | "RED" | "LDGSTS" | "LDGDEPBAR"
        | "CCTL" => "mem-other",
        "BAR" | "DEPBAR" | "MEMBAR" | "ERRBAR" | "WARPSYNC" | "BSSY" | "BSYNC" | "BRA" | "BRX"
        | "JMP" | "CALL" | "RET" | "EXIT" | "BPT" | "YIELD" | "NANOSLEEP" | "KILL" | "BREAK"
        | "NOP" => "control",
        "S2R" | "CS2R" | "S2UR" => "special-reg",
        b if b.starts_with('U') => "uniform",
        _ => "other",
    }
}

/// Ops whose result latency is variable: they must set a write
/// scoreboard when they produce a register (sanity check of the decode).
fn variable_latency(base: &str) -> bool {
    matches!(
        base,
        "LDG" | "LDS" | "LDSM" | "LDL" | "S2R" | "MUFU" | "SHFL" | "LD" | "ATOM" | "ATOMS"
    )
}

/// Fixed-latency ALU ops: must NOT need a write scoreboard.
fn fixed_latency(base: &str) -> bool {
    matches!(
        base,
        "FFMA" | "FADD" | "FMUL" | "IADD3" | "LOP3" | "ISETP" | "SEL" | "LEA" | "SHF" | "FSETP"
    )
}

/// Parse `nvdisasm` output and attach control bits from `ucode`.
pub(crate) fn parse_sass(text: &str, ucode: &[u8]) -> Result<Vec<Instr>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("/*") else {
            continue;
        };
        let Some((addr, rest)) = rest.split_once("*/") else {
            continue;
        };
        let Ok(addr) = u32::from_str_radix(addr.trim(), 16) else {
            continue;
        };
        let body = rest.trim().trim_end_matches(';').trim();
        if body.is_empty() {
            continue;
        }
        let (pred, body) = if body.starts_with('@') {
            let (p, b) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
            (Some(p.to_string()), b.trim())
        } else {
            (None, body)
        };
        let (op, operands) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
        let at = addr as usize;
        let word = ucode.get(at..at + 16).with_context(|| {
            format!("instruction at 0x{addr:x} beyond UCODE ({} B)", ucode.len())
        })?;
        out.push(Instr {
            addr,
            pred,
            op: op.to_string(),
            operands: operands.trim().to_string(),
            ctrl: Ctrl::decode(word),
        });
    }
    if out.is_empty() {
        bail!("no instructions parsed from nvdisasm output");
    }
    Ok(out)
}

/// A backward-branch loop: body = instructions in [start, end].
#[derive(Clone, Debug)]
pub(crate) struct Loop {
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// Instruction index range.
    pub(crate) range: std::ops::Range<usize>,
    /// Contains no other backward branch.
    pub(crate) innermost: bool,
}

#[derive(Default, Clone, Debug)]
pub(crate) struct Mix {
    pub(crate) total: usize,
    pub(crate) by_class: BTreeMap<&'static str, usize>,
    pub(crate) by_base: BTreeMap<String, usize>,
    /// (base, width bits) -> count for memory ops.
    pub(crate) mem: BTreeMap<(String, u32), usize>,
    pub(crate) stall_sum: u64,
    /// Tensor-core multiply-adds per warp (sum of `Instr::mma_fmas`).
    pub(crate) mma_fmas: u64,
    pub(crate) waiting: usize,
    pub(crate) yields: usize,
    pub(crate) reuse_flagged: usize,
}

impl Mix {
    pub(crate) fn of(instrs: &[Instr]) -> Self {
        let mut m = Mix::default();
        for i in instrs {
            // NOP padding after EXIT is not executed; callers slice it off.
            m.total += 1;
            *m.by_class.entry(class_of(i.base())).or_default() += 1;
            *m.by_base.entry(i.base().to_string()).or_default() += 1;
            if matches!(
                i.base(),
                "LDG" | "STG" | "LDS" | "STS" | "LDSM" | "LDL" | "STL"
            ) {
                *m.mem
                    .entry((i.base().to_string(), i.mem_width()))
                    .or_default() += 1;
            }
            m.stall_sum += u64::from(i.ctrl.stall);
            m.mma_fmas += i.mma_fmas();
            m.waiting += usize::from(i.ctrl.wait != 0);
            m.yields += usize::from(i.ctrl.yield_bit == 0);
            m.reuse_flagged += usize::from(i.ctrl.reuse != 0);
        }
        m
    }

    pub(crate) fn count(&self, base: &str) -> usize {
        self.by_base.get(base).copied().unwrap_or(0)
    }

    pub(crate) fn class(&self, class: &str) -> usize {
        self.by_class.get(class).copied().unwrap_or(0)
    }

    /// FFMA + HMMA (+ packed-half FMA) share of all instructions.
    pub(crate) fn math(&self) -> usize {
        self.count("FFMA") + self.count("HMMA") + self.count("HFMA2")
    }

    pub(crate) fn math_density(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.math() as f64 / self.total as f64
        }
    }

    /// Bytes per thread moved by one execution of these instructions.
    pub(crate) fn bytes(&self, base: &str) -> u64 {
        self.mem
            .iter()
            .filter(|((b, _), _)| b == base)
            .map(|((_, w), n)| u64::from(*w / 8) * *n as u64)
            .sum()
    }

    /// "LDG: 4x128 2x32" style width breakdown.
    pub(crate) fn widths(&self, base: &str) -> String {
        let parts: Vec<String> = self
            .mem
            .iter()
            .filter(|((b, _), _)| b == base)
            .map(|((_, w), n)| format!("{n}x{w}"))
            .collect();
        if parts.is_empty() {
            "0".into()
        } else {
            parts.join(" ")
        }
    }

    pub(crate) fn int_overhead(&self) -> usize {
        self.class("int") + self.class("uniform")
    }
}

/// The full census of one kernel.
pub(crate) struct Census {
    pub(crate) instrs: Vec<Instr>,
    /// Instructions up to and including the last EXIT/BRA-to-self
    /// (trailing NOP padding excluded).
    pub(crate) live: usize,
    pub(crate) loops: Vec<Loop>,
    pub(crate) main_loop: Option<usize>,
}

impl Census {
    pub(crate) fn new(instrs: Vec<Instr>) -> Self {
        // Live code ends at the final `BRA` to itself that follows EXIT
        // (the canonical tail), else at the last EXIT.
        let last_exit = instrs.iter().rposition(|i| i.base() == "EXIT");
        let live = match last_exit {
            Some(e) => {
                let tail = instrs[e + 1..]
                    .iter()
                    .position(|i| i.branch_target() == Some(i.addr))
                    .map(|p| e + 1 + p + 1)
                    .unwrap_or(e + 1);
                tail.min(instrs.len())
            }
            None => instrs.len(),
        };
        let index_of = |addr: u32| instrs.iter().position(|i| i.addr >= addr);
        let mut loops: Vec<Loop> = instrs[..live]
            .iter()
            .enumerate()
            .filter_map(|(idx, i)| {
                let t = i.branch_target()?;
                (t < i.addr).then(|| {
                    let s = index_of(t).unwrap_or(0);
                    Loop {
                        start: t,
                        end: i.addr,
                        range: s..idx + 1,
                        innermost: true,
                    }
                })
            })
            .collect();
        let spans: Vec<(u32, u32)> = loops.iter().map(|l| (l.start, l.end)).collect();
        for l in &mut loops {
            l.innermost = !spans
                .iter()
                .any(|&(s, e)| (s, e) != (l.start, l.end) && s >= l.start && e <= l.end);
        }
        // Main loop: the innermost loop doing the most math; ties go to
        // the leanest body.  Kernels with runtime-selected paths (the
        // row GEMVs compile normed-A x vector/scalar-B K-loops) have
        // several equal-math loops, and the leanest is the plain vector
        // path a plain dispatch executes.  `ML_ISA_LOOP=<n>` overrides.
        let main_loop = loops
            .iter()
            .enumerate()
            .filter(|(_, l)| l.innermost)
            .max_by_key(|(_, l)| {
                let m = Mix::of(&instrs[l.range.clone()]);
                (m.math(), std::cmp::Reverse(m.total))
            })
            .map(|(i, _)| i);
        Self {
            instrs,
            live,
            loops,
            main_loop,
        }
    }

    pub(crate) fn whole(&self) -> Mix {
        Mix::of(&self.instrs[..self.live])
    }

    pub(crate) fn loop_mix(&self, l: &Loop) -> Mix {
        Mix::of(&self.instrs[l.range.clone()])
    }

    /// Control-bit decode sanity: (var-lat ops producing a register with
    /// a write scoreboard, var-lat ops producing a register), (fixed-lat
    /// ops with a write scoreboard, fixed-lat ops), (trailing NOP pad
    /// decoding to the canonical padding control word, trailing NOPs).
    /// Expected: fixed-lat 0/n; var-lat close to n/n — the compiler
    /// leaves the scoreboard off a load whose result is covered by a
    /// later same-kind load's scoreboard (LDS completes in order).
    pub(crate) fn sanity(&self) -> [(usize, usize); 3] {
        let mut var = (0, 0);
        let mut fixed = (0, 0);
        let mut nops = (0, 0);
        for (idx, i) in self.instrs.iter().enumerate() {
            let b = i.base();
            // Only register-producing forms (first operand is a register).
            let writes_reg = i.operands.starts_with('R') && !i.operands.starts_with("RZ");
            if variable_latency(b) && writes_reg {
                var.1 += 1;
                var.0 += usize::from(i.ctrl.wbar != 7);
            }
            if fixed_latency(b) {
                fixed.1 += 1;
                fixed.0 += usize::from(i.ctrl.wbar != 7);
            }
            // Only the padding after the live region: alignment NOPs
            // inside the code legitimately carry stall counts.
            if b == "NOP" && idx >= self.live {
                nops.1 += 1;
                nops.0 += usize::from(i.ctrl.render() == "B------:R-:W-:Y:S00");
            }
        }
        [var, fixed, nops]
    }

    /// Latency cover inside one warp: for every `base` load in the loop
    /// that sets a write scoreboard, the distance to the first later
    /// instruction (wrapping around the loop body once) that waits on
    /// that scoreboard, as (instructions, static issue cycles).  Small
    /// distances mean the load's latency is exposed unless other warps
    /// cover it.
    pub(crate) fn use_distances(&self, l: &Loop, base: &str) -> Vec<(usize, u64)> {
        let body = &self.instrs[l.range.clone()];
        let n = body.len();
        let mut out = Vec::new();
        for (i, ins) in body.iter().enumerate() {
            if ins.base() != base || ins.ctrl.wbar == 7 {
                continue;
            }
            let bit = 1u8 << ins.ctrl.wbar;
            let mut cycles = u64::from(ins.ctrl.stall);
            for d in 1..=n {
                let next = &body[(i + d) % n];
                if next.ctrl.wait & bit != 0 {
                    out.push((d, cycles));
                    break;
                }
                cycles += u64::from(next.ctrl.stall);
            }
        }
        out
    }

    /// Main-loop body as annotated SASS (address, control, text).
    pub(crate) fn loop_listing(&self, l: &Loop) -> String {
        let mut out = String::new();
        for i in &self.instrs[l.range.clone()] {
            let _ = writeln!(
                out,
                "/*{:04x}*/ [{}] {}{} {};",
                i.addr,
                i.ctrl.render(),
                i.pred
                    .as_deref()
                    .map(|p| format!("{p} "))
                    .unwrap_or_default(),
                i.op,
                i.operands
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(lo: u64, hi: u64) -> [u8; 16] {
        let mut w = [0u8; 16];
        w[..8].copy_from_slice(&lo.to_le_bytes());
        w[8..].copy_from_slice(&hi.to_le_bytes());
        w
    }

    #[test]
    fn decodes_known_control_words() {
        // `IMAD.MOV.U32 R68, RZ, RZ, c[0x0][0x638]` from a GA104 kernel:
        // hi = 0x000fe200078e00ff -> B------:R-:W-:-:S01.
        let c = Ctrl::decode(&word(0x00018e00ff447624, 0x000fe200078e00ff));
        assert_eq!(c.render(), "B------:R-:W-:-:S01");
        // CuAssembler's SM7x/8x NOP pad: control 0x7e0 at bit 105.
        let pad = Ctrl::decode(&word(0x7918, 0x7e0u64 << 41));
        assert_eq!(pad.render(), "B------:R-:W-:Y:S00");
        // Round-trip a busy word: wait {0,5}, R0, W5, yield, S05.
        let code: u64 = (0b100001 << 11) | (0 << 8) | (5 << 5) | 5;
        let c = Ctrl::decode(&word(0, code << 41));
        assert_eq!(c.render(), "B0----5:R0:W5:Y:S05");
    }

    #[test]
    fn load_use_distance_wraps_within_the_loop() {
        let text = "\
        /*0000*/                   FFMA R8, R4, R5, R8 ;
        /*0010*/                   LDG.E R4, [R2.64] ;
        /*0020*/                   IADD3 R2, R2, 0x4, RZ ;
        /*0030*/               @P0 BRA 0x0 ;
        /*0040*/                   EXIT ;
";
        let mut ucode = vec![0u8; 0x50];
        for (i, chunk) in ucode.chunks_mut(16).enumerate() {
            // LDG sets W1 (S02); the FFMA at the loop head waits on it.
            let ctrl: u64 = match i {
                1 => (7 << 8) | (1 << 5) | 2,
                0 => (0b10 << 11) | (7 << 8) | (7 << 5) | 1,
                _ => (7 << 8) | (7 << 5) | 3,
            };
            chunk[8..].copy_from_slice(&(ctrl << 41).to_le_bytes());
        }
        let c = Census::new(parse_sass(text, &ucode).unwrap());
        let l = &c.loops[c.main_loop.unwrap()];
        // LDG (S02) -> IADD3 (S03) -> BRA (S03) -> wraps to the FFMA.
        assert_eq!(c.use_distances(l, "LDG"), vec![(3, 8)]);
    }

    #[test]
    fn parses_nvdisasm_lines_and_loops() {
        let text = "\
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   LDG.E.128.CONSTANT R4, [R2.64] ;
        /*0020*/                   FFMA R8, R4, R5, R8 ;
        /*0030*/               @P0 BRA 0x10 ;
        /*0040*/                   EXIT ;
        /*0050*/                   BRA 0x50;
        /*0060*/                   NOP;
";
        let mut ucode = vec![0u8; 0x70];
        // S2R and LDG set write scoreboard W0; everything else none.
        for (i, chunk) in ucode.chunks_mut(16).enumerate() {
            let ctrl: u64 = if i <= 1 {
                (7 << 8) | 1
            } else {
                (7 << 8) | (7 << 5) | 1
            };
            chunk[8..].copy_from_slice(&(ctrl << 41).to_le_bytes());
        }
        let instrs = parse_sass(text, &ucode).unwrap();
        assert_eq!(instrs.len(), 7);
        assert_eq!(instrs[1].mem_width(), 128);
        assert_eq!(instrs[3].pred.as_deref(), Some("@P0"));
        let c = Census::new(instrs);
        assert_eq!(c.live, 6);
        assert_eq!(c.loops.len(), 1);
        let l = &c.loops[c.main_loop.unwrap()];
        assert_eq!((l.start, l.end), (0x10, 0x30));
        let m = c.loop_mix(l);
        assert_eq!(m.total, 3);
        assert_eq!(m.count("FFMA"), 1);
        assert_eq!(m.widths("LDG"), "1x128");
        let [var, fixed, _] = c.sanity();
        assert_eq!(var, (2, 2));
        assert_eq!(fixed, (0, 1));
    }
}
