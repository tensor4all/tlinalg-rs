use super::*;

#[test]
fn recycled_buffers_come_back() {
    let mut ws = RecyclingWorkspace::default();
    let first: Vec<f64> = Workspace::<f64>::acquire_zeroed(&mut ws, 8);
    let ptr = first.as_ptr();
    Workspace::<f64>::release(&mut ws, first);
    let again: Vec<MaybeUninit<f64>> = Workspace::<f64>::acquire_uninit(&mut ws, 4);
    assert_eq!(again.as_ptr().cast::<f64>(), ptr);
    assert_eq!(again.len(), 4);
}
