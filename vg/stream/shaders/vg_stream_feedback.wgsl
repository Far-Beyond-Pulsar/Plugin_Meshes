// Virtual geometry streaming: GPU->CPU page feedback (compute stages only; needs
// the read-write vg_feedback binding). Include after vg_stream.wgsl.

const VG_FB_COUNT: u32 = 0u;
const VG_FB_CAPACITY: u32 = 1u;
const VG_FB_BITS_WORDS: u32 = 2u;
const VG_FB_HEADER_WORDS: u32 = 4u;

/// Record that a non-resident page is wanted. Deduplicated by a bitset: the
/// first caller per page per readback appends it to the request list.
fn vg_request_page(global_page: u32) {
    let bits_base = VG_FB_HEADER_WORDS + atomicLoad(&vg_feedback[VG_FB_CAPACITY]);
    let bit = 1u << (global_page & 31u);
    let word = bits_base + (global_page >> 5u);
    // Bits only accumulate until the ordered readback/reset. Avoid an RMW
    // for every instance of a shared page. Racing first readers still use
    // atomicOr's old value to elect exactly one request-list writer.
    if (atomicLoad(&vg_feedback[word]) & bit) != 0u { return; }
    let old = atomicOr(&vg_feedback[word], bit);
    if ((old & bit) == 0u) {
        let i = atomicAdd(&vg_feedback[VG_FB_COUNT], 1u);
        if (i < atomicLoad(&vg_feedback[VG_FB_CAPACITY])) {
            atomicStore(&vg_feedback[VG_FB_HEADER_WORDS + i], global_page);
        }
    }
}

/// Record that a resident page was needed (keeps it young in the LRU).
fn vg_touch_page(global_page: u32) {
    let cap = atomicLoad(&vg_feedback[VG_FB_CAPACITY]);
    let words = atomicLoad(&vg_feedback[VG_FB_BITS_WORDS]);
    let bit = 1u << (global_page & 31u);
    let word = VG_FB_HEADER_WORDS + cap + words + (global_page >> 5u);
    if (atomicLoad(&vg_feedback[word]) & bit) == 0u {
        atomicOr(&vg_feedback[word], bit);
    }
}

/// Record that `global_page` is wanted by the cut: resident pages are touched,
/// non-resident ones requested. `urgent` = the cut needs it now (it would
/// refine into it); otherwise it is a prefetch, scheduled after every urgent
/// request.
fn vg_demand_page(global_page: u32, urgent: bool) {
    if (vg_residency[global_page] != VG_PAGE_NOT_RESIDENT) {
        vg_touch_page(global_page);
        return;
    }
    if (urgent) {
        let cap = atomicLoad(&vg_feedback[VG_FB_CAPACITY]);
        let words = atomicLoad(&vg_feedback[VG_FB_BITS_WORDS]);
        let bit = 1u << (global_page & 31u);
        let word = VG_FB_HEADER_WORDS + cap + 2u * words + (global_page >> 5u);
        if (atomicLoad(&vg_feedback[word]) & bit) == 0u {
            atomicOr(&vg_feedback[word], bit);
        }
    }
    vg_request_page(global_page);
}

/// Resolve a page for drawing: returns its slot and touches it, or requests
/// it and returns VG_PAGE_NOT_RESIDENT (draw the resident parent instead).
fn vg_use_or_request_page(global_page: u32) -> u32 {
    let slot = vg_residency[global_page];
    if (slot == VG_PAGE_NOT_RESIDENT) {
        vg_request_page(global_page);
    } else {
        vg_touch_page(global_page);
    }
    return slot;
}
