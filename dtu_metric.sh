basedir=/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/gaussian-splatting/output/sa/dtu_3/
# all_id=${basedir}scan30
for scan_id in scan30 scan34 scan41 scan45 scan82 scan103 scan38 scan21 scan40 scan55 scan63 scan31 scan8 scan110 scan114
do
    python metrics_dtu.py -m $basedir$scan_id
done
# python metrics_dtu.py -m "$all_id"
