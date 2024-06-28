
export CUDA_VISIBLE_DEVICES=4

python tools/train_src.py --config-path st.regda.2potsdam \
  --align-domain 1

python tools/init_prototypes.py --config-path st.regda.2potsdam \
  --ckpt-model log/regda/2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/regda/2potsdam/src/prototypes_best.pth \
  --stage 1

python tools/train_align_reg.py --config-path st.regda.2potsdam \
  --ckpt-model log/regda/2potsdam/src/Potsdam_best.pth \
  --ckpt-proto log/regda/2potsdam/src/prototypes_best.pth \
  --align-domain 1 --refine-label 1 --sam-refine --percent 0.5

python tools/init_prototypes.py --config-path st.regda.2potsdam \
  --ckpt-model log/regda/2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/regda/2potsdam/align/prototypes_best.pth \
  --stage 2

python tools/train_ssl_reg.py --config-path st.regda.2potsdam \
  --ckpt-model log/regda/2potsdam/align/Potsdam_best.pth \
  --ckpt-proto log/regda/2potsdam/align/prototypes_best.pth \
  --gen 1 --refine-label 1 --sam-refine --percent 0.5
