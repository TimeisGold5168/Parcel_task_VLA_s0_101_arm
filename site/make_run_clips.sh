#!/usr/bin/env bash
# Page clips from the BSP robot runs (bsp/runs/<run>/ep<n>_cams.mp4),
# real time (no speed-up), 960x480, H.264. A label shows when a hand
# helps (times checked by eye on the wrist camera).
#   bash outputs/portfolio/site/make_run_clips.sh
set -euo pipefail
cd "$(dirname "$0")/videos"
R=../../bsp/runs
H5=$R/diffusion_bsp_b_parcel_hard_20261111_184111
H8=$R/diffusion_bsp_b_parcel_hard_20261111_190455
S8=$R/smolvla_bsp_b_parcel_hard_20261111_191335
ENC="-an -c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p
     -movflags +faststart"

# label TEXT FROM TO: drawtext filter shown from FROM to TO s
label() {
  echo "drawtext=text='$1':x=12:y=12:fontsize=24:fontcolor=white:\
box=1:boxcolor=0xb91c1c@0.85:boxborderw=8:enable='between(t,$2,$3)'"
}

poster() {
  local d
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1.mp4")
  ffmpeg -v error -y -ss "$(python3 -c "print($d * 0.4)")" -i "$1.mp4" \
    -frames:v 1 -q:v 5 "$1.jpg"
}

# 5-parcel heap, Diffusion-BSP, episodes 0-4 back to back (86 s).
# Hand help: ep 3 at 5-10.5 s (clip 50-55.5 s), ep 4 at 3-10 s (clip
# 71-78 s). The hand pushes the box closer to the gripper.
ffmpeg -v error -y $(for e in 0 1 2 3 4; do echo -i $H5/ep${e}_cams.mp4; done) \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1,scale=960:-2,\
$(label 'Hand help' 50 55.5),$(label 'Hand help' 71 78)" \
  $ENC heap5_diffusion_bsp.mp4
poster heap5_diffusion_bsp

# Flip: ep 3 from the grasp (12.5 s) to the end. The hand help is
# before the clip.
ffmpeg -v error -y -ss 12.5 -i $H5/ep3_cams.mp4 -vf scale=960:-2 \
  $ENC heap5_flip.mp4
poster heap5_flip

# 8-parcel heap, Diffusion-BSP ep 3: far-edge hover, hand at 22-29 s.
ffmpeg -v error -y -i $H8/ep3_cams.mp4 \
  -vf "scale=960:-2,$(label 'Hand help' 22 29)" $ENC heap8_far_edge.mp4
poster heap8_far_edge

# 8-parcel heap, SmolVLA-BSP ep 1: one move scatters the tray.
ffmpeg -v error -y -i $S8/ep1_cams.mp4 -vf scale=960:-2 \
  $ENC heap8_scatter.mp4
poster heap8_scatter

for f in heap5_diffusion_bsp heap5_flip heap8_far_edge heap8_scatter; do
  printf "%-22s %ss\n" $f \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 $f.mp4)"
done
