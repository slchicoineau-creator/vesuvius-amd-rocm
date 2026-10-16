# LASAGNA-GATE16b step 1: why did fp32_taps fail? (2026-10-13, partner, Beast; diagnosis, not a gate)

Question: GATE16 showed stock bf16 error 1x, fp32_taps 2.06x (R-hat) vs the fp32 REF, although fp32_taps is the "stock math". Which precision step makes the error?
All runs: card 1 via gpu-share, bf16 autocast, ROCBLAS_USE_HIPBLASLT=1, the real model/crop of GATE16. Scripts and raw JSON are in this folder.

## What the modes do (conv3d_fast.py, read at 66e28cf0)
| mode | input / weight / bias | accumulation | output |
|---|---|---|---|
| stock (MIOpen naive kernel `..._ushort_double_ushort`) | bf16 / bf16 / bf16 | in-kernel (unknown, see below) | bf16 |
| fp32_taps | rounded to bf16, then held in fp32 | each tap conv2d in fp32, taps + bias summed in fp32 | ONE rounding to bf16 |
| acc32_bias | bf16 (autocast) | each tap conv2d in bf16 (output of a tap is rounded to bf16), taps + bias summed in fp32 | one rounding |
| taps | bf16 | taps summed in bf16 (rounded after each add), bias added in bf16 | bf16 |

## Result 1: per layer (diag_layer_bf16in.json; 3 gate layers x 2 real tiles; input = the bf16 activations stock really feeds the layer; 48^3 centre block vs an fp64 conv)
Mean |error| / mean |exact output|, floor = bf16 rounding of the exact result of the bf16-rounded operands (RNE(R64r)):
- fp32_taps = the floor (1.3-1.6e-3) in every layer, and 99.99-100 % of its outputs equal RNE(exact). It is the correctly rounded result.
- stock 1.7-2.0e-3: only 62-78 % of outputs equal RNE(exact), the rest are 1 bf16 ulp off. So stock accumulates in something less exact than a correctly rounded double (the kernel name does not mean a double-accumulate-then-round result), error 26-43 % above the floor.
- acc32_bias 1.4-2.2e-3 (below stock in 5 of 6 layer/tile cases, above stock in one: conv1 on tile 1, 2.20 vs 1.95e-3; 62-91 % equal RNE). taps 2.0-2.3e-3, 0 % equal RNE (worst).
- A second run with fp32 inputs (diag_layer.json) gives the same ordering; the exact-rounded fp32_taps is also best there.
So at the single-layer level fp32_taps is the MOST accurate mode and the premise "stock is unusually accurate (double accumulation)" is false for these layers.

## Result 2: end to end, which layer and which mode (diag_e2e.json; tiles k = 0,1,2 of the GATE16 set; sum |out - REF| vs stock = 1)
Reproduces the cached GATE16 numbers exactly (STOCK and FAST arrays bit-identical to the cache, so REF/STOCK can be reused).
| arm | tile 0 | tile 1 | tile 2 |
|---|---|---|---|
| stock | 1.00 | 1.00 | 1.00 |
| fp32_taps L1 only (`stages.0.blocks.0.conv1`) | 2.56 | 1.00 | 2.78 |
| fp32_taps L2 only | 0.90 | 1.00 | 0.92 |
| fp32_taps L3 only | 1.00 | 1.07 | 1.00 |
| fp32_taps all 3 | 2.03 | 1.07 | 2.34 |
| taps all 3 | 1.47 | 1.23 | 1.48 |
| acc32_bias all 3 | 0.47 | 1.11 | 0.31 |
The doubling comes from ONE layer, L1 (encoder stage 0 conv1). L2 and L3 in fp32_taps are neutral or better.

## Result 3: why a per-layer-better layer is worse end to end (diag_errmap.py, diag_chan.json)
- The extra error is a coherent offset, not noise: on tile 0 / 2 the signed mean output error of fp32_taps is -0.022 / -0.052 on the largest channels (REF std 0.25) where stock has +0.001 / +0.013; per-channel |err| is 1.4-2.9x stock in all 8 channels. Correlation of the two arms' errors is ~0 (0.08, -0.06) - they are different error fields.
- At L1's own output the signed error is tiny and similar for both (|signed|/std <= 1.6e-2 per channel, stock and fp32_taps alike; diag_chan.json), so the offset is made downstream: L1's rounding pattern is amplified by the rest of the net. I did NOT isolate the downstream mechanism (candidates: normalisation of low-variance channels, deep-encoder averaging of a rounding pattern that correct rounding makes coherent where stock's 1-ulp disagreements dither it). That is a hypothesis, not a finding.
- Consequence: the per-layer error against an exact reference does NOT predict the end-to-end error against the fp32 model on this network. A mode cannot be chosen from the per-layer floor alone.

## Conclusion for step 2
- The root cause of the GATE16 FAIL is L1 (not L2/L3) and a downstream coherent offset, not the quality of the layer arithmetic (fp32_taps is the correctly rounded layer). fp32_taps will not be re-run.
- Candidate with a principled reason AND no L1 penalty: acc32_bias (fp32 sum of taps and bias, one final rounding; per-layer error mostly between the floor and stock (below stock in 5 of 6 cases); end to end 0.47 / 1.11 / 0.31 x stock on 3 tiles). taps has no reason (worst per layer, worse than stock end to end on all 3 tiles).
- Caveat on that evidence: the 3 end-to-end tiles are tiles 0-2 of the 16 the gate will use, so the choice is partly informed by gate tiles. GATE16b therefore pre-registers ONE mode, uses the unchanged 95 % bound, and says so. A PASS then means "no worse than stock on these 16 tiles", a weaker claim than a PASS on fresh tiles.
