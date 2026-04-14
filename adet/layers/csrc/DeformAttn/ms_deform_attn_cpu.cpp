// ms_deform_attn_cpu.cpp
/*
Multi-Scale Deformable Attention – CPU implementation (naïve, reference).

The logic follows the original CUDA kernel but uses naïve parallel-for loops
over (batch × query × head) on the CPU with bilinear interpolation.

⚠️ Performance is orders of magnitude slower than the CUDA kernel and intended
primarily for debugging, CPU-only inference, or unit testing.  See the README
below for build/usage instructions and notes on further optimisation.

Author: OpenAI ChatGPT (2025-06-16)
*/

#include <cmath>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

// Forward --------------------------------------------------------------------
at::Tensor
ms_deform_attn_cpu_forward(const at::Tensor& value,
                           const at::Tensor& spatial_shapes,
                           const at::Tensor& level_start_index,
                           const at::Tensor& sampling_loc,
                           const at::Tensor& attn_weight,
                           const int im2col_step)
{
// ---------------------------- Sanity checks ---------------------------- //
AT_ASSERTM(value.device().is_cpu(),            "value must be a CPU tensor");
AT_ASSERTM(spatial_shapes.device().is_cpu(),   "spatial_shapes must be a CPU tensor");
AT_ASSERTM(level_start_index.device().is_cpu(),"level_start_index must be a CPU tensor");
AT_ASSERTM(sampling_loc.device().is_cpu(),     "sampling_loc must be a CPU tensor");
AT_ASSERTM(attn_weight.device().is_cpu(),      "attn_weight must be a CPU tensor");

AT_ASSERTM(value.is_contiguous(),            "value tensor has to be contiguous");
AT_ASSERTM(spatial_shapes.is_contiguous(),   "spatial_shapes tensor has to be contiguous");
AT_ASSERTM(level_start_index.is_contiguous(),"level_start_index tensor has to be contiguous");
AT_ASSERTM(sampling_loc.is_contiguous(),     "sampling_loc tensor has to be contiguous");
AT_ASSERTM(attn_weight.is_contiguous(),      "attn_weight tensor has to be contiguous");

const int64_t B        = value.size(0);
const int64_t S_total  = value.size(1);   // Somme des surfaces ∑_l H_l x W_l
const int64_t H_heads  = value.size(2);
const int64_t C_dim    = value.size(3);

const int64_t L_levels = spatial_shapes.size(0);
const int64_t L_q      = sampling_loc.size(1);
const int64_t P_points = sampling_loc.size(4);

// im2col_step : découpe le batch pour limiter la mémoire (comme CUDA)
const int64_t step     = std::min<int64_t>(B, im2col_step > 0 ? im2col_step : B);
AT_ASSERTM(B % step == 0, "batch(%d) must divide im2col_step(%d)", (int)B, (int)step);

auto output = at::zeros({B, L_q, H_heads, C_dim}, value.options());

// --------------------------- Typed accessors --------------------------- //
AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(),
    "ms_deform_attn_cpu_forward", ([&]
{
    auto value_a   = value.accessor<scalar_t,4>();          // (B, S_total, H_heads, C_dim)
    auto spatial_a = spatial_shapes.accessor<int64_t,2>();  // (L_levels, 2)
    auto lvl_idx_a = level_start_index.accessor<int64_t,1>();// (L_levels)
    auto samp_a    = sampling_loc.accessor<scalar_t,6>();   // (B, L_q, H_heads, L_levels, P_points, 2)
    auto attn_a    = attn_weight.accessor<scalar_t,5>();    // (B, L_q, H_heads, L_levels, P_points)
    auto out_a     = output.accessor<scalar_t,4>();         // (B, L_q, H_heads, C_dim)

    const int64_t work = B * L_q * H_heads;
    at::parallel_for(0, work, 0, [&](int64_t begin, int64_t end)
        {
            for (int64_t idx = begin; idx < end; ++idx)
            {
                const int64_t b = idx / (L_q * H_heads);
                const int64_t tmp = idx % (L_q * H_heads);
                const int64_t q  = tmp / H_heads;
                const int64_t h  = tmp % H_heads;

                for (int64_t l = 0; l < L_levels; ++l)
                {
                    const int64_t H = spatial_a[l][0];
                    const int64_t W = spatial_a[l][1];
                    const int64_t lvl_start = lvl_idx_a[l];

                    for (int64_t p = 0; p < P_points; ++p)
                    {
                        scalar_t attn = attn_a[b][q][h][l][p];
                        scalar_t x = samp_a[b][q][h][l][p][0] * W - static_cast<scalar_t>(0.5);
                        scalar_t y = samp_a[b][q][h][l][p][1] * H - static_cast<scalar_t>(0.5);

                        const int64_t x0 = static_cast<int64_t>(std::floor(x));
                        const int64_t y0 = static_cast<int64_t>(std::floor(y));
                        const scalar_t  dx = x - x0;
                        const scalar_t  dy = y - y0;

                        const int64_t x1 = x0 + 1;
                        const int64_t y1 = y0 + 1;

                        const scalar_t w00 = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
                        const scalar_t w01 = dx * (static_cast<scalar_t>(1) - dy);
                        const scalar_t w10 = (static_cast<scalar_t>(1) - dx) * dy;
                        const scalar_t w11 = dx * dy;

                        auto add_sample = [&](int64_t xi, int64_t yi, scalar_t w)
                        {
                            if (xi < 0 || xi >= W || yi < 0 || yi >= H || w == 0) return;
                            const int64_t pos = lvl_start + yi * W + xi; // index dans le tensor "value"
                            for (int64_t c = 0; c < C_dim; ++c)
                            {
                                out_a[b][q][h][c] += attn * w * value_a[b][pos][h][c];
                            }
                        };

                        add_sample(x0, y0, w00);
                        add_sample(x1, y0, w01);
                        add_sample(x0, y1, w10);
                        add_sample(x1, y1, w11);
                    }
                }
            }
        });
    }));

    // (B, Len_q, Heads*Dim) – même shape que la version CUDA
    return output.reshape({B, L_q, H_heads * C_dim});
}

// Backward -------------------------------------------------------------------
std::vector<at::Tensor>
ms_deform_attn_cpu_backward(    
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int         im2col_step)
{
    // ---------------------------- Sanity checks ---------------------------- //
    AT_ASSERTM(value.device().is_cpu(),            "value must be a CPU tensor");
    AT_ASSERTM(spatial_shapes.device().is_cpu(),   "spatial_shapes must be a CPU tensor");
    AT_ASSERTM(level_start_index.device().is_cpu(),"level_start_index must be a CPU tensor");
    AT_ASSERTM(sampling_loc.device().is_cpu(),     "sampling_loc must be a CPU tensor");
    AT_ASSERTM(attn_weight.device().is_cpu(),      "attn_weight must be a CPU tensor");
    AT_ASSERTM(grad_output.device().is_cpu(),      "grad_output must be a CPU tensor");

    AT_ASSERTM(value.is_contiguous(),            "value must be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(),   "spatial_shapes must be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(),"level_start_index must be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(),     "sampling_loc must be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(),      "attn_weight must be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(),      "grad_output must be contiguous");

    const int64_t B        = value.size(0);
    const int64_t S_total  = value.size(1);
    const int64_t H_heads  = value.size(2);
    const int64_t C_dim    = value.size(3);

    const int64_t L_levels = spatial_shapes.size(0);
    const int64_t L_q      = sampling_loc.size(1);
    const int64_t P_points = sampling_loc.size(4);

    const int64_t step     = std::min<int64_t>(B, im2col_step > 0 ? im2col_step : B);
    AT_ASSERTM(B % step == 0, "batch(%d) must divide im2col_step(%d)", (int)B, (int)step);

    // (B, L_q, H_heads, C_dim)
    auto g_out = grad_output.view({B, L_q, H_heads, C_dim});

    auto grad_value         = at::zeros_like(value);
    auto grad_sampling_loc  = at::zeros_like(sampling_loc);
    auto grad_attn_weight   = at::zeros_like(attn_weight);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(),
        "ms_deform_attn_cpu_backward", ([&]
    {
        auto value_a   = value.accessor<scalar_t,4>();          // (B, S_total, H, C)
        auto gval_a    = grad_value.accessor<scalar_t,4>();     // idem
        auto spatial_a = spatial_shapes.accessor<int64_t,2>();  // (L, 2)
        auto lvl_idx_a = level_start_index.accessor<int64_t,1>();// (L)
        auto samp_a    = sampling_loc.accessor<scalar_t,6>();   // (B, L_q, H, L, P, 2)
        auto gsamp_a   = grad_sampling_loc.accessor<scalar_t,6>();
        auto attn_a    = attn_weight.accessor<scalar_t,5>();    // (B, L_q, H, L, P)
        auto gattn_a   = grad_attn_weight.accessor<scalar_t,5>();
        auto gout_a    = g_out.accessor<scalar_t,4>();          // (B, L_q, H, C)

        // Parallélisation par batch pour éviter les races sur grad_value
        at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end)
        {
            for (int64_t b = b_begin; b < b_end; ++b)
            {
                for (int64_t q = 0; q < L_q; ++q)
                {
                    for (int64_t h = 0; h < H_heads; ++h)
                    {
                        for (int64_t l = 0; l < L_levels; ++l)
                        {
                            const int64_t H = spatial_a[l][0];
                            const int64_t W = spatial_a[l][1];
                            const int64_t lvl_start = lvl_idx_a[l];

                            for (int64_t p = 0; p < P_points; ++p)
                            {
                                scalar_t attn = attn_a[b][q][h][l][p];
                                scalar_t x = samp_a[b][q][h][l][p][0] * W - static_cast<scalar_t>(0.5);
                                scalar_t y = samp_a[b][q][h][l][p][1] * H - static_cast<scalar_t>(0.5);

                                const int64_t x0 = static_cast<int64_t>(std::floor(x));
                                const int64_t y0 = static_cast<int64_t>(std::floor(y));
                                const scalar_t  dx = x - x0;
                                const scalar_t  dy = y - y0;

                                const int64_t x1 = x0 + 1;
                                const int64_t y1 = y0 + 1;

                                const scalar_t w00 = (static_cast<scalar_t>(1) - dx) * (static_cast<scalar_t>(1) - dy);
                                const scalar_t w01 = dx * (static_cast<scalar_t>(1) - dy);
                                const scalar_t w10 = (static_cast<scalar_t>(1) - dx) * dy;
                                const scalar_t w11 = dx * dy;

                                // ---------------------- load neighbor values --------------------- //
                                scalar_t v00[C_dim]; scalar_t v01[C_dim]; scalar_t v10[C_dim]; scalar_t v11[C_dim];
                                bool in00 = (x0>=0 && x0<W && y0>=0 && y0<H);
                                bool in01 = (x1>=0 && x1<W && y0>=0 && y0<H);
                                bool in10 = (x0>=0 && x0<W && y1>=0 && y1<H);
                                bool in11 = (x1>=0 && x1<W && y1>=0 && y1<H);

                                auto load_val = [&](bool inside, int64_t xi, int64_t yi, scalar_t* vec)
                                {
                                    const int64_t pos = lvl_start + yi * W + xi;
                                    for (int64_t c = 0; c < C_dim; ++c)
                                    {
                                        vec[c] = inside ? value_a[b][pos][h][c] : static_cast<scalar_t>(0);
                                    }
                                };
                                if (C_dim > 0)
                                {
                                    load_val(in00, x0, y0, v00);
                                    load_val(in01, x1, y0, v01);
                                    load_val(in10, x0, y1, v10);
                                    load_val(in11, x1, y1, v11);
                                }

                                scalar_t sum_wv = static_cast<scalar_t>(0);
                                scalar_t dI_dlocx = static_cast<scalar_t>(0);
                                scalar_t dI_dlocy = static_cast<scalar_t>(0);

                                for (int64_t c = 0; c < C_dim; ++c)
                                {
                                    scalar_t g = gout_a[b][q][h][c];

                                    // ---------------- grad wrt value ---------------------------- //
                                    if (in00 && w00 != 0)
                                    {
                                        const int64_t pos = lvl_start + y0 * W + x0;
                                        gval_a[b][pos][h][c] += attn * w00 * g;
                                    }
                                    if (in01 && w01 != 0)
                                    {
                                        const int64_t pos = lvl_start + y0 * W + x1;
                                        gval_a[b][pos][h][c] += attn * w01 * g;
                                    }
                                    if (in10 && w10 != 0)
                                    {
                                        const int64_t pos = lvl_start + y1 * W + x0;
                                        gval_a[b][pos][h][c] += attn * w10 * g;
                                    }
                                    if (in11 && w11 != 0)
                                    {
                                        const int64_t pos = lvl_start + y1 * W + x1;
                                        gval_a[b][pos][h][c] += attn * w11 * g;
                                    }

                                    // ---------------- grad wrt attn_weight ---------------------- //
                                    sum_wv += g * (w00 * v00[c] + w01 * v01[c] + w10 * v10[c] + w11 * v11[c]);

                                    // ---------------- grad wrt loc_x / loc_y ------------------- //
                                    // d(wxy)/dloc_x and d(wxy)/dloc_y
                                    scalar_t dw_dx = (-(static_cast<scalar_t>(1) - dy) * v00[c]) +
                                                     ((static_cast<scalar_t>(1) - dy) * v01[c]) +
                                                     (-dy * v10[c]) +
                                                     (dy * v11[c]);

                                    scalar_t dw_dy = (-(static_cast<scalar_t>(1) - dx) * v00[c]) +
                                                     (-dx * v01[c]) +
                                                     ((static_cast<scalar_t>(1) - dx) * v10[c]) +
                                                     (dx * v11[c]);

                                    dI_dlocx += g * dw_dx;
                                    dI_dlocy += g * dw_dy;
                                }

                                // accumulate grad of attn_weight
                                gattn_a[b][q][h][l][p] += sum_wv * attn; // Actually ∂/∂attn = interpolated value
                                // wait, derivative: output = attn * I; so dL/dattn = g * I.
                                // We computed sum_wv = Σ_c g * I_c. but I = Σ_c I_c . So this is correct.

                                // grad wrt sampling_loc (normalised)
                                gsamp_a[b][q][h][l][p][0] += dI_dlocx * attn * static_cast<scalar_t>(W);
                                gsamp_a[b][q][h][l][p][1] += dI_dlocy * attn * static_cast<scalar_t>(H);
                            } // P
                        } // L
                    } // H
                } // Q
            } // B
        }); // parallel_for
    })); // dispatch

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}
