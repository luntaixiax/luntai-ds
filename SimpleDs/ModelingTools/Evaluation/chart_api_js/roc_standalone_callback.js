// console.log("Hello");

const anno_data = source_anno.data;
const raw_data = source_roc_test.data;
const threshold = cb_obj.value;

var vs = null;

const thresholds = raw_data['threshold'];
for (var i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= threshold) {
        const thres = thresholds[i];
        const tpr = raw_data['tpr'][i];
        const fpr = raw_data['fpr'][i];
        vs = [thres, tpr, fpr];
        break;
    }
}

anno_data['threshold'][0] =  vs[0];
anno_data['tpr'][0] = vs[1];
anno_data['fpr'][0] = vs[2];
source_anno.change.emit();