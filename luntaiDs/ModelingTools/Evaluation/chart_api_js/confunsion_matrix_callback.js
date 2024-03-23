//console.log("Hello");

const origin_data = origin_source.data;
const error_data = error_source.data;
const raw_data = raw_source.data;
const threshold = cb_obj.value;

var v_original = null;
var v_error = null;

const thresholds = raw_data['threshold'];
for (var i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= threshold) {
        const fp = raw_data['fp'][i];
        const tp = raw_data['tp'][i];
        const tn = raw_data['tn'][i];
        const fn = raw_data['fn'][i];

        const tpr = raw_data['tpr'][i];
        const fnr = 1 - tpr;
        const fpr = raw_data['fpr'][i];
        const tnr = 1 - fpr;

        v_original = [fp, tp, tn, fn];
        v_error = [fpr, 0, 0, fnr];
        break;
    }
}

const origin_values = origin_data['value'];
for (var i = 0; i < origin_values.length; i++) {
    origin_values[i] = v_original[i];
}
const error_values = error_data['value'];
for (var i = 0; i < error_values.length; i++) {
    error_values[i] = v_error[i];
}

origin_source.change.emit();
error_source.change.emit();