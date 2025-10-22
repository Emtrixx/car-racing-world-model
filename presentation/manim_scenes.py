from pathlib import Path
import numpy as np

# Manim community edition
from manim import (
    Scene, VGroup, Rectangle, RoundedRectangle, Arrow, Brace,
    Text, FadeIn, FadeOut, Write, Create, Transform,
    Indicate, LaggedStart, GrowArrow, CurvedArrow, SurroundingRectangle,
    BLUE, GREEN, YELLOW, RED, PURPLE, ORANGE, TEAL, WHITE, BLACK,
    DOWN, UP, RIGHT, LEFT
)


class ColorTheme:
    bg = BLACK
    text = WHITE
    block_fill = {
        "input": TEAL,
        "encode": GREEN,
        "core": YELLOW,
        "latent": ORANGE,
        "heads": PURPLE,
        "output": BLUE,
        "mask": RED,
    }
    block_opacity = 0.15
    block_stroke = 0.9


def block(label: str, width=3.6, height=1.4, kind="input"):
    card = RoundedRectangle(width=width, height=height, corner_radius=0.12,
                            color=ColorTheme.block_fill.get(kind, BLUE),
                            stroke_opacity=ColorTheme.block_stroke,
                            fill_opacity=ColorTheme.block_opacity)
    txt = Text(label, font_size=28, color=ColorTheme.text)
    grp = VGroup(card, txt)
    txt.move_to(card.get_center())
    return grp


def labeled_arrow(src_mob, dst_mob, label=None, buff=0.2):
    a = Arrow(src_mob.get_right(), dst_mob.get_left(), buff=buff)
    if label:
        t = Text(label, font_size=24, color=ColorTheme.text).next_to(a, UP, buff=0.1)
        return a, t
    return a


def token_grid(n=16, cell=0.18, label=None, color=BLUE, opacity=0.35):
    """Return a VGroup representing a square grid of tokens."""
    size = int(np.sqrt(n))
    squares = VGroup()
    for _ in range(size * size):
        sq = Rectangle(
            width=cell,
            height=cell,
            color=color,
            stroke_width=1.0,
            fill_opacity=opacity,
        )
        squares.add(sq)
    squares.arrange_in_grid(rows=size, cols=size, buff=0.03)
    if label:
        label_m = Text(label, font_size=22, color=ColorTheme.text).next_to(squares, UP, buff=0.15)
        return VGroup(squares, label_m)
    return squares


class TokenizationScene(Scene):
    """Visualize VQ-VAE tokenization concept: frame -> tokens -> embeddings + grid pos enc."""

    def construct(self):
        self.camera.background_color = ColorTheme.bg

        frame = block("CarRacing Frame", kind="input", width=4.5, height=2.7)
        tokens = token_grid(n=16, label="VQ-VAE Tokens (4x4)")
        embed = block("Codebook Embedding\n(d -> d_model)", kind="encode")
        pos = block("Add 2D Grid Pos Enc", kind="encode")

        frame.to_edge(LEFT).shift(UP * 0.5)
        tokens.next_to(frame, RIGHT, buff=1.0)
        embed.next_to(tokens, RIGHT, buff=1.2)
        pos.next_to(embed, RIGHT, buff=1.0)

        self.play(FadeIn(frame))
        self.play(Create(tokens))
        arr1, lbl1 = labeled_arrow(frame, tokens, label="VQ-VAE index")
        self.play(Create(arr1), Write(lbl1))
        self.play(FadeIn(embed))
        arr2, lbl2 = labeled_arrow(tokens, embed, label="token -> vector")
        self.play(Create(arr2), Write(lbl2))
        self.play(FadeIn(pos))
        arr3, lbl3 = labeled_arrow(embed, pos, label="+ grid pos")
        self.play(Create(arr3), Write(lbl3))

        title = Text("Tokenization & Positional Encoding", font_size=34, color=ColorTheme.text).to_edge(UP)
        self.play(Write(title))
        self.wait(1.5)


class GRUWorldModelScene(Scene):
    """Animated overview of the GRU world model pipeline (src/world_model.py)."""

    def construct(self):
        self.camera.background_color = ColorTheme.bg

        # Inputs
        obs = block("Observation Tokens\nB x T x N", kind="input")
        act = block("Actions\nB x T x A", kind="input")
        h0 = block("Hidden State h0\nL x B x d", kind="input")
        inputs = VGroup(obs, act, h0).arrange(DOWN, buff=0.5).to_edge(LEFT)

        # Encoding
        enc = block("Token Embedding +\nGrid Pos + MLP", kind="encode")
        act_enc = block("Action Embedding", kind="encode")
        cat = block("Concat [obs_emb, act_emb]", kind="encode", width=4.2)
        encs = VGroup(enc, act_enc, cat).arrange(DOWN, buff=0.5).next_to(inputs, RIGHT, buff=1.0)

        # Core
        core = block("Stacked GRU (batch_first)\nDeterministic state h_t", kind="core", width=4.8)
        core.next_to(encs, RIGHT, buff=1.2)

        # Stochastic branch
        stoch = block("MLP -> (μ, log σ²)\nrsample z_t", kind="latent")
        comb = block("Concat [h_t, z_t]", kind="latent")
        stoch.next_to(core, RIGHT, buff=1.0)
        comb.next_to(stoch, RIGHT, buff=0.8)

        # Heads
        tok = block("Linear -> token logits\nB x T x (N x K)", kind="heads", width=4.3)
        rew = block("Reward Head", kind="heads")
        done = block("Done Head", kind="heads")
        heads = VGroup(tok, rew, done).arrange(DOWN, buff=0.5)
        heads.next_to(comb, RIGHT, buff=1.2)

        # Build
        title = Text("GRU World Model (Deterministic + Stochastic)", font_size=34, color=ColorTheme.text).to_edge(UP)
        self.play(Write(title))
        self.play(FadeIn(inputs))

        # Flow: obs -> enc
        self.play(FadeIn(enc), FadeIn(act_enc))
        a1, l1 = labeled_arrow(obs, enc, label="embed + MLP")
        a2, l2 = labeled_arrow(act, act_enc, label="linear")
        self.play(Create(a1), Write(l1), Create(a2), Write(l2))

        self.play(FadeIn(cat))
        a3 = labeled_arrow(enc, cat)
        a4 = labeled_arrow(act_enc, cat)
        self.play(Create(a3), Create(a4))

        self.play(FadeIn(core))
        a5, l5 = labeled_arrow(cat, core, label="[obs_emb, act_emb]")
        a6, l6 = labeled_arrow(h0, core, label="initial h")
        self.play(Create(a5), Write(l5), Create(a6), Write(l6))

        self.play(FadeIn(stoch))
        a7, l7 = labeled_arrow(core, stoch, label="predict μ, log σ²")
        self.play(Create(a7), Write(l7))

        self.play(FadeIn(comb))
        a8 = labeled_arrow(stoch, comb)
        a9 = labeled_arrow(core, comb)
        self.play(Create(a8), Create(a9))

        self.play(FadeIn(heads))
        self.play(Create(labeled_arrow(comb, tok)), Create(labeled_arrow(comb, rew)), Create(labeled_arrow(comb, done)))
        self.wait(1.5)


class TransformerWorldModelScene(Scene):
    """Animated overview of the Transformer world model with T-BTF (src/transformer_world_model.py)."""

    def construct(self):
        self.camera.background_color = ColorTheme.bg

        title = Text(
            "Transformer World Model",
            font_size=34,
            color=ColorTheme.text,
        ).to_edge(UP)
        self.play(Write(title))

        subtitle = Text("Memory construction", font_size=30, color=ColorTheme.text)
        subtitle.next_to(title, DOWN, buff=0.6)
        self.play(Write(subtitle))

        history_len = 3
        grid_size = 4
        num_tokens = grid_size * grid_size

        token_steps = VGroup()
        step_labels = []
        for idx in range(history_len):
            grid = token_grid(n=num_tokens, cell=0.2, color=BLUE)
            grid.scale(0.85)
            offset = history_len - 1 - idx
            step_name = f"t-{offset}" if offset > 0 else "t"
            label = Text(step_name, font_size=20, color=ColorTheme.text)
            step_group = VGroup(grid, label).arrange(DOWN, buff=0.1)
            token_steps.add(step_group)
            step_labels.append(label)

        token_steps.arrange(RIGHT, buff=0.55)
        token_steps.move_to(np.array([0.0, -0.3, 0.0]))

        self.play(LaggedStart(*[FadeIn(step, shift=UP * 0.15) for step in token_steps], lag_ratio=0.15))
        brace_hist = Brace(token_steps, DOWN, color=ColorTheme.text)
        brace_hist_text = Text("VQ-VAE latent token history", font_size=22, color=ColorTheme.text)
        brace_hist_text.next_to(brace_hist, DOWN, buff=0.12)
        brace_hist_text.set_x(token_steps.get_center()[0])
        self.play(Create(brace_hist), Write(brace_hist_text))

        annotation = Text("Add learnable 2D grid positional encoding", font_size=24, color=ColorTheme.text)
        annotation.next_to(token_steps, UP, buff=0.35)
        annotation.set_x(token_steps.get_center()[0])
        self.play(Write(annotation))
        self.play(LaggedStart(*[Indicate(step[0], scale_factor=1.05) for step in token_steps], lag_ratio=0.2))

        # Flatten each timestep row for interleaving
        rows_targets = []
        base_left = token_steps.get_left() + RIGHT * 0.9
        for idx, step in enumerate(token_steps):
            row = step[0].copy()
            row.arrange(RIGHT, buff=0.04)
            row.scale(0.72)
            row.move_to(base_left + DOWN * (idx * 0.7))
            rows_targets.append(row)

        flatten_note = Text("Flatten tokens per timestep", font_size=24, color=ColorTheme.text).move_to(annotation)
        self.play(Transform(annotation, flatten_note), FadeOut(brace_hist), FadeOut(brace_hist_text))
        self.play(*[Transform(token_steps[idx][0], rows_targets[idx]) for idx in range(history_len)])
        self.play(
            *[step_labels[idx].animate.next_to(token_steps[idx][0], LEFT, buff=0.18) for idx in range(history_len)])

        # Action embeddings per timestep
        token_width = token_steps[0][0][0].width
        token_height = token_steps[0][0][0].height
        action_blocks = VGroup()
        for idx in range(history_len):
            action_box = Rectangle(width=token_width, height=token_height,
                                   color=ColorTheme.block_fill["encode"],
                                   stroke_opacity=0.9,
                                   fill_opacity=0.3)
            action_text = Text("a", font_size=18, color=ColorTheme.text).move_to(action_box.get_center())
            action = VGroup(action_box, action_text)
            action.next_to(token_steps[idx][0], RIGHT, buff=0.2)
            action_blocks.add(action)

        action_note = Text("Linear action embeddings", font_size=24, color=ColorTheme.text).move_to(annotation)
        self.play(Transform(annotation, action_note))
        self.play(LaggedStart(*[FadeIn(action) for action in action_blocks], lag_ratio=0.15))

        interleave_note = Text("Interleave tokens with action embeddings", font_size=24, color=ColorTheme.text).move_to(
            annotation)
        self.play(Transform(annotation, interleave_note))

        interleave_rows = VGroup()
        for idx in range(history_len):
            combined = VGroup(token_steps[idx][0], action_blocks[idx])
            interleave_rows.add(combined)
        interleave_brace = Brace(interleave_rows, RIGHT, color=ColorTheme.text)
        self.play(Create(interleave_brace))

        temporal_note = Text("Add sinusoidal temporal positional encoding", font_size=24,
                             color=ColorTheme.text).move_to(annotation)
        self.play(Transform(annotation, temporal_note))
        self.play(
            LaggedStart(*[Indicate(row, scale_factor=1.03, color=ORANGE) for row in interleave_rows], lag_ratio=0.2))

        # Build single interleaved memory sequence
        sequence_elements = []
        for idx in range(history_len):
            sequence_elements.extend(token_steps[idx][0])
            sequence_elements.append(action_blocks[idx])
        memory_sequence = VGroup(*sequence_elements)
        self.play(memory_sequence.animate.arrange(RIGHT, buff=0.06, center=False))
        memory_sequence.scale(0.9)
        memory_sequence.center().shift(DOWN * 0.4)
        self.play(FadeOut(interleave_brace), FadeOut(VGroup(*step_labels)), FadeOut(annotation))

        memory_box = SurroundingRectangle(memory_sequence, color=ColorTheme.block_fill["encode"], buff=0.25)
        memory_label = Text("Interleaved memory (tokens + actions)", font_size=24, color=ColorTheme.text)
        memory_label.next_to(memory_box, DOWN, buff=0.22)
        token_legend_box = Rectangle(width=token_width, height=token_height, color=BLUE, fill_opacity=0.35)
        token_legend_text = Text("latent token", font_size=18, color=ColorTheme.text).next_to(token_legend_box, RIGHT,
                                                                                              buff=0.15)
        action_legend_box = Rectangle(width=token_width, height=token_height, color=ColorTheme.block_fill["encode"],
                                      fill_opacity=0.35)
        action_legend_text = Text("action embedding", font_size=18, color=ColorTheme.text).next_to(action_legend_box,
                                                                                                   RIGHT, buff=0.15)
        legend = VGroup(VGroup(token_legend_box, token_legend_text),
                        VGroup(action_legend_box, action_legend_text)).arrange(RIGHT, buff=0.5)
        legend.next_to(memory_label, DOWN, buff=0.18)

        self.play(Create(memory_box), Write(memory_label), FadeIn(legend))
        self.wait(1.0)

        memory_group = VGroup(memory_sequence, memory_box, memory_label, legend)
        memory_group.to_edge(UP, buff=1.1)

        # Transition to decoder queries (single-step inference)
        next_subtitle = Text("Single-step decoder queries", font_size=30, color=ColorTheme.text)
        next_subtitle.next_to(memory_group, DOWN, buff=0.6)
        self.play(Transform(subtitle, next_subtitle))

        query_grid = token_grid(n=num_tokens, cell=0.22, color=PURPLE)
        query_grid.scale(0.85)
        global_rect = Rectangle(width=token_width * 1.1, height=token_height * 1.1,
                                color=PURPLE, fill_opacity=0.28, stroke_opacity=0.9)
        global_text = Text("g", font_size=18, color=ColorTheme.text).move_to(global_rect.get_center())
        global_token = VGroup(global_rect, global_text)
        queries_group = VGroup(query_grid, global_token).arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
        queries_group.next_to(next_subtitle, DOWN, buff=1.2)
        queries_group.set_x(0.0)
        self.play(FadeIn(queries_group, shift=DOWN * 0.4))

        queries_label = Text("Decoder queries (tokens + global)", font_size=23, color=ColorTheme.text)
        queries_label.next_to(queries_group, DOWN, buff=0.2)
        queries_label.set_x(queries_group.get_center()[0])
        self.play(Write(queries_label))

        stage_note = Text("Add 2D spatial + temporal encodings", font_size=24, color=ColorTheme.text)
        stage_note.next_to(queries_group, UP, buff=0.3)
        stage_note.set_x(queries_group.get_center()[0])
        self.play(Write(stage_note))
        highlight_anims = [Indicate(sq, color=ORANGE, scale_factor=1.04) for sq in query_grid]
        highlight_anims.append(Indicate(global_rect, color=PURPLE, scale_factor=1.03))
        self.play(LaggedStart(*highlight_anims, lag_ratio=0.06, run_time=1.2))

        # Self-attention among query tokens
        self_attn_caption = Text("Self-attention between query tokens", font_size=24, color=ColorTheme.text).move_to(stage_note)
        self.play(Transform(stage_note, self_attn_caption))

        self_attn_box = SurroundingRectangle(query_grid, color=ColorTheme.block_fill["mask"], buff=0.28)
        pair_indices = [(0, 5), (6, 13), (2, 10)]
        self_attn_arrows = VGroup()
        for i, j in pair_indices:
            arrow = CurvedArrow(
                query_grid[i].get_center(),
                query_grid[j].get_center(),
                angle=0.5,
                tip_length=0.1,
                color=ColorTheme.block_fill["mask"],
                stroke_width=4.0,
            )
            self_attn_arrows.add(arrow)
        self.play(Create(self_attn_box))
        self.play(LaggedStart(*[Create(arrow) for arrow in self_attn_arrows], lag_ratio=0.2, run_time=2.0))
        self.wait(2.0)
        self.play(FadeOut(self_attn_arrows), FadeOut(self_attn_box))

        # Cross-attention glimpses from memory to queries
        cross_caption = Text("Cross-attention: queries attend to memory", font_size=24, color=ColorTheme.text).move_to(stage_note)
        self.play(Transform(stage_note, cross_caption))
        self.play(Indicate(memory_sequence, scale_factor=1.01))
        cross_specs = [
            (query_grid[2], memory_sequence[3], 3.5),
            (query_grid[9], memory_sequence[len(memory_sequence) // 2], 5.0),
            (global_rect, memory_sequence[-2], 4.2),
        ]
        cross_arrows = VGroup()
        for src, dst, width in cross_specs:
            src_mob = src[0] if isinstance(src, VGroup) else src
            dst_mob = dst[0] if isinstance(dst, VGroup) else dst
            arrow = Arrow(
                src_mob.get_top(),
                dst_mob.get_bottom(),
                buff=0.12,
                color=ORANGE,
                stroke_width=width,
                tip_length=0.18,
            )
            cross_arrows.add(arrow)
        self.play(LaggedStart(*[GrowArrow(ar) for ar in cross_arrows], lag_ratio=0.2, run_time=2.0))
        self.wait(2.0)
        self.play(FadeOut(cross_arrows))

        self.play(FadeOut(stage_note), FadeOut(queries_label))
        self.wait(1.5)


class TransformerAttentionScene(Scene):
    """
    Visualize attention maps (self/cross) for one layer and head.
    If a file exists at presentation/assets/attention_maps.npz, it is used.
    Expected keys: 'self' (list of arrays), 'cross' (list of arrays)
    with shapes [B, H, Q, K]. If not found, random maps are displayed.
    """

    def construct(self):
        self.camera.background_color = ColorTheme.bg
        title = Text("Transformer Attention Maps", font_size=36, color=ColorTheme.text).to_edge(UP)
        self.play(Write(title))

        path = Path("presentation/assets/attention_maps.npz")
        if path.exists():
            data = np.load(path, allow_pickle=True)
            # stored as object arrays -> unwrap
            cross_list = list(data.get('cross', []))
            self_list = list(data.get('self', []))
            attn_cross = cross_list[0][0] if len(cross_list) else None  # [heads, Q, K]
            attn_self = self_list[0][0] if len(self_list) else None
        else:
            # fallback to random demo maps
            heads, Q, K_mem, K_self = 4, 20, 60, 20
            attn_cross = np.random.dirichlet(np.ones(K_mem), size=(heads, Q)).reshape(heads, Q, K_mem)
            attn_self = np.random.dirichlet(np.ones(K_self), size=(heads, Q)).reshape(heads, Q, K_self)

        def heatmap(attn, width=6.0, height=3.3, title_tx=""):
            if attn is None:
                return Text("No attention available", color=ColorTheme.text)
            H, Q, K = attn.shape
            # normalize 0..1
            arr = attn / (attn.max() + 1e-8)
            cell_w = width / K
            cell_h = height / Q
            grid = VGroup()
            for i in range(Q):
                for j in range(K):
                    val = arr[0, i, j]  # first head for clarity
                    col = BLUE.interpolate(RED, val)
                    r = Rectangle(width=cell_w, height=cell_h, color=col, fill_opacity=0.85, stroke_opacity=0.0)
                    r.move_to(np.array([
                        (j - K / 2 + 0.5) * cell_w,
                        (Q / 2 - i - 0.5) * cell_h,
                        0.0
                    ]))
                    grid.add(r)
            border = Rectangle(width=width, height=height, color=WHITE, stroke_opacity=0.6)
            label = Text(title_tx, font_size=28, color=ColorTheme.text).next_to(border, UP, buff=0.2)
            return VGroup(border, grid, label)

        cross_vis = heatmap(attn_cross, title_tx="Cross-Attention (Queries vs. History)")
        self_vis = heatmap(attn_self, title_tx="Self-Attention (Queries vs. Queries)")
        both = VGroup(self_vis, cross_vis).arrange(DOWN, buff=0.8).scale(0.9)
        both.move_to(0.5 * DOWN)

        self.play(FadeIn(both))
        self.wait(1.5)
