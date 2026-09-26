#!/usr/bin/env bash
# Compile-only ISA probes of the M1 generator configs + literal twins.
set -u
BIN=$1; OUT=$2; shift 2
for k in "$@"; do
  case $k in *m64n64|*_t2|*_n1|*_n4|*_n9|*_n10) c=q512,1,512,2048,2048,f16w;;
             *m128n64|*_t4|*_n2|*_n5) c=gu512,1,512,5632,2048,f16w;;
             *m64n128|*_t3|*_n3|*_n6) c=q512,1,512,2048,2048,f16w;;
             *_n11) c=q512,1,512,2048,2048,f16w;;
             *) c=s2k,1,2048,2048,2048,f16w;; esac
  ML_KERNEL=$k ML_ISA_OUT=$OUT RUST_LOG=warn "$BIN" isa $c >> "$OUT/isa.log" 2>&1 || echo "isa $k failed"
done
