
search_dir='./output/video/*'

for entry in $search_dir; do
    python ./VIBE/demo.py --vid_file $entry --output_folder ./output/joint/vibe/
done