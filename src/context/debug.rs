use std::ffi::CStr;

use ash::vk;

pub(super) unsafe extern "system" fn debug_callback(
    _sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _typ: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    let msg_ptr = unsafe { (*data).p_message };
    if !msg_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(msg_ptr) }.to_string_lossy();
        eprintln!("[vulkan] {msg}");
    }
    vk::FALSE
}
