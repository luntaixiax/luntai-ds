const normalize_type = cb_obj.value;

if (normalize_type === 'All') {
    cur_s.data = all_s.data;
} else if (normalize_type === 'Base') {
    cur_s.data = prod_s.data;
} else {
    cur_s.data = dev_s.data;
}

cur_s.change.emit();