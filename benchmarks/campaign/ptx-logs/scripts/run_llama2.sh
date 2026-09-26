#!/usr/bin/env bash
# llama.cpp reference + token-exactness of the PTX route with a 128-token prompt (a16 path).
cd ~/ta-wt/c-ptx
L=target/release/llama_ash
M=~/models/tinyllama-1.1b-chat-v1.0.f16.gguf
IDS="1,8019,15938,3857,11776,19695,7614,15533,3452,11371,19290,7209,15128,3047,10966,18885,6804,14723,2642,10561,18480,6399,14318,2237,10156,18075,5994,13913,1832,9751,17670,5589,13508,1427,9346,17265,5184,13103,1022,8941,16860,4779,12698,617,8536,16455,4374,12293,212,8131,16050,3969,11888,19807,7726,15645,3564,11483,19402,7321,15240,3159,11078,18997,6916,14835,2754,10673,18592,6511,14430,2349,10268,18187,6106,14025,1944,9863,17782,5701,13620,1539,9458,17377,5296,13215,1134,9053,16972,4891,12810,729,8648,16567,4486,12405,324,8243,16162,4081,12000,19919,7838,15757,3676,11595,19514,7433,15352,3271,11190,19109,7028,14947,2866,10785,18704,6623,14542,2461,10380,18299,6218,14137,2056,9975,17894,5813"
echo "## generate 128-token prompt, 32 greedy tokens"
for mode in off f32 f16p f16; do
  echo "# $mode"; ML_PTX=1 ML_PTX_GEMM=$mode timeout 120 $L generate -m $M --ids "$IDS" -n 32 2>&1 | grep "generated_ids"
done
echo "## llama.cpp CUDA llama-bench (b9503)"
timeout 150 ~/models/llama-cpp-cuda/bin/llama-bench -m $M -p 512 -n 32 -fa 1 -r 5 2>&1 | grep -E "\| *llama|pp512|tg32|model"
echo "## tensor-ash pp512 alternating base / ptx f16 / ptx f16p"
for rep in 1 2; do
  for mode in off f16 f16p; do
    echo "# $mode"; ML_PTX=1 ML_PTX_GEMM=$mode timeout 120 $L bench -m $M --pp 512 --tg 32 2>&1 | grep -E "prefill timing|pp512"
  done
done
