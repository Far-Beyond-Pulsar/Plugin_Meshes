struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var accumulation: texture_2d<f32>;
@group(0) @binding(2) var revealage: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let scene = textureLoad(scene_color, pixel, 0);
    let accum = textureLoad(accumulation, pixel, 0);
    let transmittance = clamp(textureLoad(revealage, pixel, 0).r, 0.0, 1.0);
    let coverage = 1.0 - transmittance;
    let transparent_color = accum.rgb / max(accum.a, 1.0e-5);
    return vec4<f32>(mix(scene.rgb, transparent_color, coverage), scene.a);
}
