// console.log("Hello");

const anno_data = source_anno.data;
const raw_data = source_pr_test.data;
const threshold = cb_obj.value;

var vs = null;

const thresholds = raw_data['threshold'];
for (var i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= threshold) {
        const thres = thresholds[i];
        const recall = raw_data['recall'][i];
        const precision = raw_data['precision'][i];
        vs = [thres, recall, precision];
        break;
    }
}

anno_data['threshold'][0] =  vs[0];
anno_data['recall'][0] = vs[1];
anno_data['precision'][0] = vs[2];
source_anno.change.emit();