#version 430 core

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(location = 0) in vec2 position;

#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

layout (std430, binding=0) buffer gaussian_data {
    float g_data[];
};
layout (std430, binding=1) buffer gaussian_order {
    int gi[];
};

// 4D temporal parameters SSBO
// Per-gaussian layout: [time(1), duration(1), vx, vy, vz,
//   ax, ay, az (order>=2), jx, jy, jz (order>=3), sx, sy, sz (order>=4)]
layout (std430, binding=2) buffer gaussian_4d_data {
    float g_4d[];
};

// Turbo colormap LUT (256 entries x 3 floats)
layout (std430, binding=3) buffer turbo_lut {
    float lut_data[];  // 256 * 3 floats, accessed as lut_data[idx*3 + channel]
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform int render_mod;

// 4D uniforms
uniform float current_time;        // normalized t in [0, 1]
uniform int motion_order;          // 1-4
uniform int heatmap_mode;          // 0=normal, 1=velocity, 2=temporal_opa, 3=duration, 4=base_opa, 5=grad
uniform int selected_gaussian;     // -1 = none
uniform float temporal_threshold;  // default 0.01
uniform int is_4d_mode;            // 0=static PLY, 1=4D checkpoint
uniform float heatmap_min;         // scalar min for normalization
uniform float heatmap_max;         // scalar max for normalization

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;

mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    mat3 R = mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );
    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
    t.y = min(limy, max(-limy, t.y / t.z)) * t.z;
    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset)
{
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

vec3 turbo_lookup(float t_val)
{
    int idx = clamp(int(t_val * 255.0), 0, 255);
    return vec3(lut_data[idx * 3], lut_data[idx * 3 + 1], lut_data[idx * 3 + 2]);
}

void main()
{
    int boxid = gi[gl_InstanceID];
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    int start = boxid * total_dim;

    vec3 canonical_pos = get_vec3(start + POS_IDX);
    vec3 g_pos_xyz = canonical_pos;
    float g_opacity = g_data[start + OPACITY_IDX];
    float temporal_opa = 1.0;

    // === 4D mode: apply motion model and temporal opacity ===
    if (is_4d_mode == 1) {
        int dim_4d = 5 + max(0, motion_order - 1) * 3;
        int start_4d = boxid * dim_4d;

        float g_time = g_4d[start_4d];
        float g_duration_log = g_4d[start_4d + 1];
        vec3 g_velocity = vec3(g_4d[start_4d + 2], g_4d[start_4d + 3], g_4d[start_4d + 4]);

        // Temporal opacity: exp(-0.5 * (dt / max(exp(duration), 0.02))^2)
        float dt = current_time - g_time;
        float s = max(exp(g_duration_log), 0.02);
        temporal_opa = exp(-0.5 * (dt / s) * (dt / s));

        // Cull low temporal opacity
        if (temporal_opa < temporal_threshold) {
            gl_Position = vec4(-100, -100, -100, 1);
            return;
        }

        // Motion displacement: polynomial up to motion_order
        vec3 displacement = g_velocity * dt;
        if (motion_order >= 2) {
            vec3 accel = vec3(g_4d[start_4d + 5], g_4d[start_4d + 6], g_4d[start_4d + 7]);
            float dt2 = dt * dt;
            displacement += 0.5 * accel * dt2;
            if (motion_order >= 3) {
                vec3 jerk = vec3(g_4d[start_4d + 8], g_4d[start_4d + 9], g_4d[start_4d + 10]);
                displacement += 0.1666667 * jerk * dt2 * dt;
                if (motion_order >= 4) {
                    vec3 snap_v = vec3(g_4d[start_4d + 11], g_4d[start_4d + 12], g_4d[start_4d + 13]);
                    displacement += 0.0416667 * snap_v * dt2 * dt2;
                }
            }
        }

        g_pos_xyz = canonical_pos + displacement;

        // Final opacity = sigmoid(logit) * temporal_opacity
        // g_opacity is already sigmoid-activated from CPU side
        g_opacity = g_opacity * temporal_opa;
    }

    vec4 g_pos = vec4(g_pos_xyz, 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
    vec4 g_pos_screen = projection_matrix * g_pos_view;
    g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;

    // Early culling
    if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3)))) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }

    vec4 g_rot = get_vec4(start + ROT_IDX);
    vec3 g_scale = get_vec3(start + SCALE_IDX);

    mat3 cov3d = computeCov3D(g_scale * scale_modifier, g_rot);
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d = computeCov2D(g_pos_view, hfovxy_focal.z, hfovxy_focal.z,
                              hfovxy_focal.x, hfovxy_focal.y, cov3d, view_matrix);

    float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
    if (det == 0.0f)
        gl_Position = vec4(0.f, 0.f, 0.f, 0.f);

    float det_inv = 1.f / det;
    conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));

    // === Selected gaussian: enlarge quad for visibility ===
    if (boxid == selected_gaussian) {
        quadwh_scr *= 3.0;  // 3x bigger so it's easy to spot
    }

    vec2 quadwh_ndc = quadwh_scr / wh * 2;
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;

    alpha = g_opacity;

    // === Selected gaussian highlight ===
    if (boxid == selected_gaussian) {
        color = vec3(1.0, 0.2, 0.0);  // bright orange-red
        alpha = 1.0;  // fully opaque so it's always visible
        return;
    }

    // === Heatmap mode ===
    if (heatmap_mode > 0 && is_4d_mode == 1) {
        int dim_4d = 5 + max(0, motion_order - 1) * 3;
        int start_4d = boxid * dim_4d;
        float scalar = 0.0;

        if (heatmap_mode == 1) {
            // Velocity magnitude
            vec3 vel = vec3(g_4d[start_4d + 2], g_4d[start_4d + 3], g_4d[start_4d + 4]);
            scalar = length(vel);
        } else if (heatmap_mode == 2) {
            // Temporal opacity (already computed)
            scalar = temporal_opa;
        } else if (heatmap_mode == 3) {
            // Duration (exp of log-duration)
            scalar = exp(g_4d[start_4d + 1]);
        } else if (heatmap_mode == 4) {
            // Base opacity (already activated)
            scalar = g_data[start + OPACITY_IDX];
        }
        // heatmap_mode == 5 (gradient) is handled CPU-side via separate scalar upload

        // Normalize using CPU-provided min/max
        float range = heatmap_max - heatmap_min;
        float norm_val = (range > 1e-8) ? clamp((scalar - heatmap_min) / range, 0.0, 1.0) : 0.5;
        color = turbo_lookup(norm_val);
        return;
    }

    // === Depth mode ===
    if (render_mod == -1) {
        float depth = -g_pos_view.z;
        depth = depth < 0.05 ? 1 : depth;
        depth = 1 / depth;
        color = vec3(depth, depth, depth);
        return;
    }

    // === SH to color ===
    int sh_start = start + SH_IDX;
    vec3 dir = g_pos.xyz - cam_pos;
    dir = normalize(dir);
    color = SH_C0 * get_vec3(sh_start);

    if (sh_dim > 3 && render_mod >= 1) {
        float x = dir.x; float y = dir.y; float z = dir.z;
        color = color - SH_C1 * y * get_vec3(sh_start + 1*3)
                      + SH_C1 * z * get_vec3(sh_start + 2*3)
                      - SH_C1 * x * get_vec3(sh_start + 3*3);

        if (sh_dim > 12 && render_mod >= 2) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            color = color +
                SH_C2_0 * xy * get_vec3(sh_start + 4*3) +
                SH_C2_1 * yz * get_vec3(sh_start + 5*3) +
                SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_start + 6*3) +
                SH_C2_3 * xz * get_vec3(sh_start + 7*3) +
                SH_C2_4 * (xx - yy) * get_vec3(sh_start + 8*3);

            if (sh_dim > 27 && render_mod >= 3) {
                color = color +
                    SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_start + 9*3) +
                    SH_C3_1 * xy * z * get_vec3(sh_start + 10*3) +
                    SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_start + 11*3) +
                    SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_start + 12*3) +
                    SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_start + 13*3) +
                    SH_C3_5 * z * (xx - yy) * get_vec3(sh_start + 14*3) +
                    SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_start + 15*3);
            }
        }
    }
    color += 0.5f;
}
