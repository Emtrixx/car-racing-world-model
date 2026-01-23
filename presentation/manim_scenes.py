from pathlib import Path
import numpy as np

# Manim community edition
from manim import (
    Scene, VGroup, Rectangle, RoundedRectangle, Arrow, Brace, Circle, Square, Line,
    Text, MathTex, FadeIn, FadeOut, Write, Create, Transform, Uncreate,
    Indicate, LaggedStart, LaggedStartMap, GrowArrow, CurvedArrow, SurroundingRectangle, DecimalNumber,
    BLUE, BLUE_C, BLUE_D, BLUE_E, GREEN, GREEN_C, YELLOW, RED, RED_C, PURPLE, PURPLE_A, PURPLE_B,
    ORANGE, TEAL, TEAL_C, TEAL_D, WHITE, BLACK, GREY, GREY_B, GREY_D, GOLD,
    DOWN, UP, RIGHT, LEFT, ORIGIN, UR, UL, DR, DL,
    TransformFromCopy, ReplacementTransform, AnimationGroup, Succession,
    MoveToTarget, ApplyMethod, Flash, ShowPassingFlash, GrowFromCenter,
    config, DoubleArrow, DashedLine, Dot, Polygon, VMobject,
    linear, smooth, there_and_back, rate_functions,
)

from threeb_style import (
    ThreeBPalette,
    make_token_strip,
    add_caption,
    animate_positional_overlay,
    attention_flow,
    make_numeric_embedding,
    add_legend,
    next_token_bar_chart,
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


def block(label: str, width=3.6, height=1.4, kind="input", font_size=22) -> VGroup:
    card = RoundedRectangle(width=width, height=height, corner_radius=0.12,
                            color=ColorTheme.block_fill.get(kind, BLUE),
                            stroke_opacity=ColorTheme.block_stroke,
                            fill_opacity=ColorTheme.block_opacity)
    txt = Text(label, font_size=font_size, color=ColorTheme.text)
    grp = VGroup(card, txt)
    txt.move_to(card.get_center())
    return grp


def latent_node(label: str, color, *, width: float = 1.4, height: float = 0.8,
                opacity: float = 0.32, font_size: int = 24) -> VGroup:
    """Compact rounded card to represent latent vectors (h_t, z_t, e_t, a_t)."""

    card = RoundedRectangle(width=width, height=height, corner_radius=0.16, color=color,
                            stroke_opacity=0.9, fill_opacity=opacity)
    txt = Text(label, font_size=font_size, color=ColorTheme.text)
    txt.move_to(card.get_center())
    return VGroup(card, txt)


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


# =============================================================================
# TRANSFORMER WORLD MODEL - Reusable Components
# =============================================================================

# Use a consistent font to avoid spacing issues
DEFAULT_FONT = "Sans"  # or "Monospace" for fixed-width


def styled_text(text: str, font_size: int = 16, color=WHITE, weight: str = "NORMAL", **kwargs) -> Text:
    """Create text with consistent font to avoid spacing issues."""
    return Text(text, font_size=font_size, color=color, font=DEFAULT_FONT, weight=weight, **kwargs)


class TransformerColors:
    """Color scheme for Transformer World Model visualizations."""
    token_fill = BLUE_C
    token_stroke = BLUE_D
    action_fill = TEAL_C
    action_stroke = TEAL_D
    query_fill = PURPLE_A
    query_stroke = PURPLE_B
    global_token = ORANGE
    pos_encoding = YELLOW
    attention_self = YELLOW
    attention_cross = ORANGE
    mask_allowed = GREEN_C
    mask_blocked = RED_C
    embedding_gradient = (BLUE_E, BLUE_C)
    text = WHITE
    caption = GREY_B


def make_frame_placeholder(width: float = 1.5, height: float = 1.5) -> VGroup:
    """Create a placeholder for a game frame (64×64 Car Racing observation)."""
    frame = Rectangle(
        width=width, height=height,
        color=GREEN_C, fill_opacity=0.2, stroke_width=2
    )
    # Add simple track visualization
    track = Rectangle(width=width * 0.6, height=height * 0.8, color=GREY_D, fill_opacity=0.3)
    track.move_to(frame.get_center())
    # Add a simple car representation
    car = Rectangle(width=0.15, height=0.25, color=RED_C, fill_opacity=0.8)
    car.move_to(frame.get_center() + DOWN * 0.2)
    label = Text("64×64", font_size=14, color=TransformerColors.text)
    label.next_to(frame, DOWN, buff=0.1)
    return VGroup(frame, track, car, label)


def make_token_grid_indexed(
        rows: int = 4, cols: int = 4, cell_size: float = 0.4,
        color=None, show_indices: bool = False, indices: list = None
) -> VGroup:
    """Create a grid of tokens with optional index labels."""
    color = color or TransformerColors.token_fill
    grid = VGroup()
    idx = 0
    for r in range(rows):
        for c in range(cols):
            cell = Square(
                side_length=cell_size,
                color=color,
                fill_opacity=0.4,
                stroke_width=1.5
            )
            if show_indices and indices:
                idx_text = Text(str(indices[idx] % 1000), font_size=10, color=WHITE)
                idx_text.move_to(cell.get_center())
                cell = VGroup(cell, idx_text)
            grid.add(cell)
            idx += 1
    grid.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
    return grid


def make_embedding_bar(
        width: float = 2.0, height: float = 0.3, color=None,
        label: str = None, gradient: bool = True
) -> VGroup:
    """Create a horizontal bar representing an embedding vector."""
    color = color or TransformerColors.token_fill
    if gradient:
        # Create gradient effect with multiple segments
        num_segments = 12
        segments = VGroup()
        seg_width = width / num_segments
        for i in range(num_segments):
            alpha = i / (num_segments - 1)
            seg_color = BLUE_E if alpha < 0.5 else BLUE_C
            opacity = 0.3 + 0.5 * (1 - abs(alpha - 0.5) * 2)
            seg = Rectangle(
                width=seg_width, height=height,
                color=seg_color, fill_opacity=opacity, stroke_width=0
            )
            segments.add(seg)
        segments.arrange(RIGHT, buff=0)
        frame = RoundedRectangle(
            width=width, height=height + 0.04,
            corner_radius=0.06, color=color, stroke_width=1.5
        )
        frame.move_to(segments.get_center())
        bar = VGroup(segments, frame)
    else:
        bar = RoundedRectangle(
            width=width, height=height,
            corner_radius=0.06, color=color,
            fill_opacity=0.5, stroke_width=1.5
        )

    if label:
        lbl = Text(label, font_size=16, color=TransformerColors.text)
        lbl.next_to(bar, RIGHT, buff=0.15)
        return VGroup(bar, lbl)
    return bar


def make_token_sequence(
        num_tokens: int, cell_size: float = 0.35,
        token_color=None, action_color=None, include_action: bool = True,
        compact: bool = False
) -> VGroup:
    """Create a sequence of image tokens optionally followed by an action token."""
    token_color = token_color or TransformerColors.token_fill
    action_color = action_color or TransformerColors.action_fill

    sequence = VGroup()
    for i in range(num_tokens):
        cell = Square(
            side_length=cell_size,
            color=token_color,
            fill_opacity=0.45,
            stroke_width=1.5
        )
        sequence.add(cell)

    if include_action:
        action_cell = Square(
            side_length=cell_size,
            color=action_color,
            fill_opacity=0.55,
            stroke_width=2
        )
        sequence.add(action_cell)

    buff = 0.03 if compact else 0.06
    sequence.arrange(RIGHT, buff=buff)
    return sequence


def make_positional_encoding_wave(
        width: float = 4.0, height: float = 0.6, num_waves: int = 3
) -> VGroup:
    """Create a visualization of sinusoidal positional encoding."""
    waves = VGroup()
    for i in range(num_waves):
        # Create sine wave points
        num_points = 50
        points = []
        freq = (i + 1) * 2
        for j in range(num_points):
            x = -width / 2 + (j / (num_points - 1)) * width
            y = (height / 2) * np.sin(freq * np.pi * j / (num_points - 1))
            points.append([x, y, 0])

        wave = VMobject()
        wave.set_points_smoothly([np.array(p) for p in points])
        wave.set_stroke(
            color=[YELLOW, ORANGE][i % 2],
            width=2,
            opacity=0.7 - i * 0.15
        )
        waves.add(wave)

    return waves


def make_attention_matrix(
        rows: int, cols: int, cell_size: float = 0.25,
        mask_pattern: str = "full", block_size: int = 4,
        show_weights: bool = False
) -> VGroup:
    """
    Create an attention mask visualization.

    mask_pattern options:
    - "full": all cells filled (no mask)
    - "full_heatmap": all cells with variable attention weights
    - "block_diagonal": only blocks on diagonal are filled
    - "causal": lower triangular with block structure

    show_weights: if True, show varying intensities for attention weights
    """
    matrix = VGroup()
    np.random.seed(42)  # For reproducible "random" weights

    for r in range(rows):
        for c in range(cols):
            cell = Square(side_length=cell_size, stroke_width=0.5, stroke_color=GREY_D)

            # Determine if this cell should be filled based on mask pattern
            filled = False
            if mask_pattern == "full" or mask_pattern == "full_heatmap":
                filled = True
            elif mask_pattern == "block_diagonal":
                block_r = r // block_size
                block_c = c // block_size
                filled = (block_r == block_c)
            elif mask_pattern == "causal":
                block_r = r // block_size
                block_c = c // block_size
                filled = (block_c <= block_r)

            if filled:
                if mask_pattern == "full_heatmap" or show_weights:
                    # Create varying attention weights for visual interest
                    weight = np.random.uniform(0.2, 1.0)
                    # Diagonal tends to have higher weights
                    if r == c:
                        weight = np.random.uniform(0.7, 1.0)
                    cell.set_fill(TransformerColors.mask_allowed, opacity=0.3 + 0.5 * weight)
                else:
                    cell.set_fill(TransformerColors.mask_allowed, opacity=0.6)
            else:
                cell.set_fill(TransformerColors.mask_blocked, opacity=0.15)

            matrix.add(cell)

    matrix.arrange_in_grid(rows=rows, cols=cols, buff=0.02)
    return matrix


def make_decoder_layer_block(
        width: float = 5.0, height: float = 0.8,
        label: str = "Transformer Decoder Layer"
) -> VGroup:
    """Create a block representing a transformer decoder layer."""
    block = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.12, color=PURPLE,
        fill_opacity=0.2, stroke_width=2
    )
    text = Text(label, font_size=18, color=TransformerColors.text)
    text.move_to(block.get_center())
    return VGroup(block, text)


def make_prediction_head(
        width: float = 1.8, height: float = 0.6,
        label: str = "Head", color=PURPLE
) -> VGroup:
    """Create a prediction head block."""
    block = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.1, color=color,
        fill_opacity=0.3, stroke_width=2
    )
    text = Text(label, font_size=16, color=TransformerColors.text)
    text.move_to(block.get_center())
    return VGroup(block, text)


# =============================================================================
# Scene 1: Memory Construction
# =============================================================================

class TransformerWM_MemoryConstruction(Scene):
    """
    Visualizes how the Transformer World Model constructs its memory sequence
    from VQ-VAE encoded observations and actions.
    """

    def construct(self):
        self.camera.background_color = "#0f1419"

        # Scene sections
        self.show_title()
        self.show_vqvae_encoding()
        self.show_token_embedding()
        self.show_grid_positional_encoding()
        self.show_action_embedding()
        self.show_memory_assembly()
        self.show_temporal_positional_encoding()
        self.show_final_memory()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "Transformer World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 1: Memory Construction",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_vqvae_encoding(self):
        """Show how a game frame is encoded into VQ-VAE tokens."""
        # Section title
        section_title = Text(
            "Step 1: VQ-VAE Encoding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create frame placeholder
        frame = make_frame_placeholder(width=2.0, height=2.0)
        frame.move_to(LEFT * 4 + DOWN * 0.3)
        frame_label = Text("Observation", font_size=20, color=WHITE)
        frame_label.next_to(frame, UP, buff=0.2)

        self.play(FadeIn(frame, frame_label, shift=LEFT))
        self.wait(0.5)

        # Arrow and encoder block
        encoder_block = block("VQ-VAE\nEncoder", width=2.2, height=1.2, kind="encode", font_size=18)
        encoder_block.move_to(LEFT * 0.8 + DOWN * 0.3)

        arrow1 = Arrow(frame.get_right(), encoder_block.get_left(), buff=0.2, color=WHITE)

        self.play(GrowArrow(arrow1))
        self.play(FadeIn(encoder_block, shift=RIGHT))
        self.wait(0.5)

        # Create token grid with placeholder indices
        indices = [127, 43, 512, 89, 234, 67, 401, 156,
                   312, 78, 445, 23, 189, 356, 267, 98]
        token_grid = make_token_grid_indexed(
            rows=4, cols=4, cell_size=0.45,
            color=TransformerColors.token_fill,
            show_indices=True, indices=indices
        )
        token_grid.move_to(RIGHT * 3.5 + DOWN * 0.3)
        grid_label = Text("4×4 Token Grid", font_size=20, color=WHITE)
        grid_label.next_to(token_grid, UP, buff=0.2)

        arrow2 = Arrow(encoder_block.get_right(), token_grid.get_left(), buff=0.2, color=WHITE)

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(token_grid, grid_label, shift=RIGHT))

        # Add explanation caption
        caption = Text(
            "Each token is an index into a codebook of 512 learned embeddings",
            font_size=18, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP))

        self.wait(2)

        # Clean up but keep token grid for next section
        self.play(
            FadeOut(section_title, frame, frame_label, encoder_block, arrow1, arrow2, caption, grid_label),
            token_grid.animate.move_to(LEFT * 4 + UP * 1.5).scale(0.8)
        )
        self.wait(0.3)

        # Store for next section
        self.token_grid = token_grid

    def show_token_embedding(self):
        """Show token embedding process."""
        section_title = Text(
            "Step 2: Token Embedding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Add label to existing token grid
        grid_label = Text("Token Indices", font_size=16, color=WHITE)
        grid_label.next_to(self.token_grid, UP, buff=0.15)
        self.play(FadeIn(grid_label))

        # Arrow to embedding lookup
        embed_block = block("Embedding\nLookup", width=2.0, height=1.0, kind="encode", font_size=16)
        embed_block.move_to(LEFT * 0.5 + UP * 1.5)

        arrow1 = Arrow(self.token_grid.get_right(), embed_block.get_left(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow1), FadeIn(embed_block, shift=RIGHT))

        # Create 16 embedding bars (4x4 arranged)
        embed_grid = VGroup()
        for _ in range(16):
            bar = make_embedding_bar(width=0.8, height=0.18, gradient=True)
            embed_grid.add(bar)
        embed_grid.arrange_in_grid(rows=4, cols=4, buff=0.08)
        embed_grid.move_to(RIGHT * 3 + UP * 1.5)
        embed_label = styled_text("256-dim Embeddings", font_size=16, color=WHITE)
        embed_label.next_to(embed_grid, UP, buff=0.15)

        arrow2 = Arrow(embed_block.get_right(), embed_grid.get_left(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow2))
        self.play(LaggedStart(
            *[FadeIn(bar, shift=RIGHT * 0.5) for bar in embed_grid],
            lag_ratio=0.05
        ))
        self.play(FadeIn(embed_label))

        # Show projection to 512-dim
        proj_block = block("Linear\nProjection", width=1.8, height=0.9, kind="core", font_size=14)
        proj_block.move_to(RIGHT * 0.5 + DOWN * 0.8)

        arrow3 = Arrow(embed_grid.get_bottom(), proj_block.get_top(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow3), FadeIn(proj_block, shift=DOWN))

        # Final 512-dim embeddings
        final_embed_grid = VGroup()
        for _ in range(16):
            bar = make_embedding_bar(width=1.2, height=0.22, gradient=True)
            final_embed_grid.add(bar)
        final_embed_grid.arrange_in_grid(rows=4, cols=4, buff=0.1)
        final_embed_grid.move_to(RIGHT * 0.5 + DOWN * 2.5)
        final_label = Text("512-dim Token Embeddings", font_size=16, color=WHITE)
        final_label.next_to(final_embed_grid, DOWN, buff=0.15)

        arrow4 = Arrow(proj_block.get_bottom(), final_embed_grid.get_top(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow4))
        self.play(LaggedStart(
            *[FadeIn(bar, shift=DOWN * 0.3) for bar in final_embed_grid],
            lag_ratio=0.03
        ))
        self.play(FadeIn(final_label))

        self.wait(1.5)

        # Clean up
        self.play(FadeOut(
            section_title, grid_label, self.token_grid, arrow1, embed_block,
            arrow2, embed_grid, embed_label, arrow3, proj_block, arrow4, final_label
        ))

        # Keep final embeddings and move to center-left
        self.play(final_embed_grid.animate.move_to(LEFT * 3.5 + UP * 0.5).scale(0.9))
        self.wait(0.3)

        self.token_embeddings = final_embed_grid

    def show_grid_positional_encoding(self):
        """Show learned grid positional encoding being added."""
        section_title = styled_text(
            "Step 3: Grid Positional Encoding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move token embeddings to left and scale down for better layout
        self.play(self.token_embeddings.animate.move_to(LEFT * 5 + DOWN * 0.5).scale(0.7))

        # Label for token embeddings
        embed_label = styled_text("Token\nEmbeddings", font_size=12, color=WHITE)
        embed_label.next_to(self.token_embeddings, UP, buff=0.15)
        self.play(FadeIn(embed_label))

        # Create positional encoding grid (different color per position) - smaller
        pos_grid = VGroup()
        colors = [YELLOW, GOLD, ORANGE, RED_C] * 4
        for i in range(16):
            bar = RoundedRectangle(
                width=0.7, height=0.14,
                corner_radius=0.04,
                color=colors[i % len(colors)],
                fill_opacity=0.5,
                stroke_width=1.5
            )
            pos_grid.add(bar)
        pos_grid.arrange_in_grid(rows=4, cols=4, buff=0.06)
        pos_grid.move_to(LEFT * 1.4 + DOWN * 0.5)
        pos_label = styled_text("Position\nEmbeddings", font_size=12, color=WHITE)
        pos_label.next_to(pos_grid, UP, buff=0.15)

        self.play(FadeIn(pos_grid, pos_label, shift=DOWN))

        # Plus sign - positioned between token and pos embeddings
        plus = styled_text("+", font_size=32, color=WHITE)
        plus.move_to(LEFT * 3.1 + DOWN * 0.5)
        self.play(FadeIn(plus))

        # Equals sign - positioned after pos embeddings
        equals = styled_text("=", font_size=32, color=WHITE)
        equals.move_to(RIGHT * 0.5 + DOWN * 0.5)
        self.play(FadeIn(equals))

        # Result grid - to the right
        result_grid = VGroup()
        for i in range(16):
            bar = make_embedding_bar(width=0.7, height=0.14, gradient=True)
            bar[1].set_stroke(colors[i % len(colors)], width=2)
            result_grid.add(bar)
        result_grid.arrange_in_grid(rows=4, cols=4, buff=0.06)
        result_grid.move_to(RIGHT * 2.4 + DOWN * 0.5)
        result_label = styled_text("Position-Encoded\nEmbeddings", font_size=12, color=WHITE)
        result_label.next_to(result_grid, UP, buff=0.15)

        self.play(FadeIn(result_grid, result_label, shift=RIGHT))

        # Animate the addition
        self.play(
            Indicate(pos_grid, color=YELLOW, scale_factor=1.05),
            run_time=1
        )

        # Caption
        caption = styled_text(
            "Each of the 16 grid positions has a unique learnable embedding",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP))

        self.wait(1.5)

        # Clean up
        self.play(FadeOut(
            section_title, embed_label, self.token_embeddings, pos_grid, pos_label,
            plus, equals, result_label, caption
        ))
        self.play(result_grid.animate.move_to(LEFT * 4 + UP * 1).scale(1.2))

        self.positioned_embeddings = result_grid

    def show_action_embedding(self):
        """Show action embedding process."""
        section_title = Text(
            "Step 4: Action Embedding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Show action vector
        action_vector = VGroup()
        action_labels = ["steer", "gas", "brake"]
        action_values = ["0.3", "0.8", "0.0"]
        for i, (label, val) in enumerate(zip(action_labels, action_values)):
            cell = Square(side_length=0.6, color=TEAL_C, fill_opacity=0.4, stroke_width=2)
            val_text = Text(val, font_size=14, color=WHITE)
            label_text = Text(label, font_size=10, color=GREY_B)
            val_text.move_to(cell.get_center())
            label_text.next_to(cell, DOWN, buff=0.05)
            action_vector.add(VGroup(cell, val_text, label_text))
        action_vector.arrange(RIGHT, buff=0.1)
        action_vector.move_to(LEFT * 0.5 + DOWN * 1)

        action_title = Text("Action (3-dim)", font_size=16, color=WHITE)
        action_title.next_to(action_vector, UP, buff=0.2)

        self.play(FadeIn(action_vector, action_title, shift=UP))

        # Linear embedding block
        embed_block = block("Linear\nEmbedding", width=1.8, height=0.9, kind="input", font_size=14)
        embed_block.move_to(RIGHT * 2.5 + DOWN * 1)

        arrow1 = Arrow(action_vector.get_right(), embed_block.get_left(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow1), FadeIn(embed_block, shift=RIGHT))

        # Action embedding result
        action_embed = make_embedding_bar(width=1.6, height=0.35, color=TEAL_C, gradient=False)
        action_embed[0].set_fill(TEAL_C, opacity=0.5)
        action_embed.move_to(RIGHT * 5 + DOWN * 1)
        embed_label = Text("512-dim", font_size=14, color=WHITE)
        embed_label.next_to(action_embed, DOWN, buff=0.1)

        arrow2 = Arrow(embed_block.get_right(), action_embed.get_left(), buff=0.15, color=WHITE)

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(action_embed, embed_label, shift=RIGHT))

        self.wait(1)

        # Clean up but keep action embedding
        self.play(FadeOut(
            section_title, action_vector, action_title, embed_block,
            arrow1, arrow2, embed_label
        ))
        self.play(action_embed.animate.move_to(LEFT * 4 + DOWN * 1.5).scale(0.7))

        self.action_embedding = action_embed

    def show_memory_assembly(self):
        """Show how tokens and action are assembled into memory sequence."""
        section_title = Text(
            "Step 5: Memory Sequence Assembly",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Show current positioned embeddings and action
        pos_label = Text("16 Image Tokens", font_size=14, color=WHITE)
        pos_label.next_to(self.positioned_embeddings, UP, buff=0.1)
        action_label = Text("1 Action", font_size=14, color=WHITE)
        action_label.next_to(self.action_embedding, UP, buff=0.1)

        self.play(FadeIn(pos_label, action_label))

        # Animate flattening the grid to a sequence
        flat_sequence = make_token_sequence(16, cell_size=0.28, include_action=True, compact=True)
        flat_sequence.move_to(ORIGIN + DOWN * 0.5)

        # Create brace
        brace = Brace(flat_sequence, DOWN, color=WHITE)
        brace_text = Text("17 elements per timestep", font_size=14, color=GREY_B)
        brace_text.next_to(brace, DOWN, buff=0.1)

        self.play(
            ReplacementTransform(self.positioned_embeddings.copy(), flat_sequence[:-1]),
            ReplacementTransform(self.action_embedding.copy(), flat_sequence[-1]),
            run_time=1.5
        )
        self.play(FadeIn(brace, brace_text))

        # Show multiple timesteps
        multi_timestep_label = Text(
            "For H timesteps in history:",
            font_size=18, color=WHITE
        ).move_to(UP * 2)
        self.play(
            FadeOut(pos_label, action_label, self.positioned_embeddings, self.action_embedding),
            Write(multi_timestep_label)
        )

        # Create multiple timestep sequences
        timesteps = VGroup()
        timestep_labels = []
        for t in range(3):
            seq = make_token_sequence(16, cell_size=0.22, include_action=True, compact=True)
            t_label = Text(f"t={t}", font_size=12, color=GREY_B)
            timesteps.add(seq)
            timestep_labels.append(t_label)

        timesteps.arrange(DOWN, buff=0.3)
        timesteps.move_to(ORIGIN)

        for i, (seq, label) in enumerate(zip(timesteps, timestep_labels)):
            label.next_to(seq, LEFT, buff=0.2)

        self.play(
            FadeOut(flat_sequence, brace, brace_text),
            run_time=0.5
        )

        self.play(
            LaggedStart(*[FadeIn(seq, shift=DOWN) for seq in timesteps], lag_ratio=0.2),
            LaggedStart(*[FadeIn(lbl) for lbl in timestep_labels], lag_ratio=0.2)
        )

        # Show concatenation arrow
        concat_arrow = Arrow(ORIGIN + RIGHT * 2.5, ORIGIN + RIGHT * 4, color=WHITE)
        concat_text = Text("Concatenate", font_size=14, color=WHITE)
        concat_text.next_to(concat_arrow, UP, buff=0.1)

        self.play(GrowArrow(concat_arrow), FadeIn(concat_text))

        # Show final long sequence
        final_seq = make_token_sequence(12, cell_size=0.15, include_action=False, compact=True)
        dots = Text("...", font_size=20, color=WHITE)
        final_display = VGroup(final_seq, dots)
        final_display.arrange(RIGHT, buff=0.1)
        final_display.move_to(RIGHT * 5.5)

        final_label = Text("H × 17 total", font_size=12, color=GREY_B)
        final_label.next_to(final_display, DOWN, buff=0.1)

        self.play(FadeIn(final_display, final_label, shift=RIGHT))

        self.wait(1.5)

        # Store and clean up
        self.memory_timesteps = timesteps
        self.timestep_labels = timestep_labels

        self.play(FadeOut(
            section_title, multi_timestep_label, concat_arrow, concat_text,
            final_display, final_label, timesteps, *timestep_labels
        ))

        # Center the timesteps
        # self.play(VGroup(timesteps, *timestep_labels).animate.move_to(LEFT * 2.5))

    def show_temporal_positional_encoding(self):
        """Show sinusoidal temporal positional encoding added to concatenated memory."""
        section_title = styled_text(
            "Step 6: Temporal Positional Encoding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # First show the concatenated memory sequence
        # self.play(FadeOut(self.memory_timesteps, *self.timestep_labels))

        # Create concatenated memory visualization
        concat_memory = VGroup()
        for t in range(3):
            for i in range(5):  # 5 tokens per timestep shown
                color = TransformerColors.token_fill if i < 4 else TransformerColors.action_fill
                cell = Square(
                    side_length=0.22,
                    color=color,
                    fill_opacity=0.5 if i < 4 else 0.6,
                    stroke_width=1
                )
                concat_memory.add(cell)
            if t < 2:
                sep = styled_text("|", font_size=14, color=GREY_D)
                concat_memory.add(sep)

        concat_memory.arrange(RIGHT, buff=0.03)
        concat_memory.move_to(LEFT * 2 + DOWN * 0.5)

        mem_label = styled_text("Concatenated Memory Sequence", font_size=14, color=WHITE)
        mem_label.next_to(concat_memory, UP, buff=0.2)

        self.play(FadeIn(concat_memory, mem_label, shift=UP))

        # Create positional encoding visualization
        pos_enc_wave = make_positional_encoding_wave(width=3.0, height=0.4)
        pos_enc_wave.move_to(RIGHT * 3.5 + UP * 1)

        wave_label = styled_text("Sinusoidal Encoding", font_size=14, color=YELLOW)
        wave_label.next_to(pos_enc_wave, UP, buff=0.15)

        self.play(FadeIn(wave_label))
        self.play(Create(pos_enc_wave), run_time=1.5)

        # Show addition with arrow
        add_arrow = Arrow(pos_enc_wave.get_bottom(), concat_memory.get_right() + RIGHT * 0.3,
                          color=YELLOW, buff=0.1, stroke_width=3)
        add_label = styled_text("+ add", font_size=12, color=YELLOW)
        add_label.next_to(add_arrow, RIGHT, buff=0.1)

        self.play(GrowArrow(add_arrow), FadeIn(add_label))

        # Show formula
        formula = MathTex(
            r"\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)",
            font_size=20, color=GREY_B
        ).move_to(RIGHT * 3.5 + DOWN * 1.5)

        self.play(FadeIn(formula, shift=UP))

        # Animate the encoding being applied
        self.play(
            Indicate(concat_memory, color=YELLOW, scale_factor=1.03),
            run_time=1
        )

        caption = styled_text(
            "Position encoding allows the model to understand temporal order",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP))

        self.wait(1.5)

        # Clean up
        self.play(FadeOut(
            section_title, pos_enc_wave, wave_label, add_arrow, add_label,
            formula, caption, concat_memory, mem_label
        ))

    def show_final_memory(self):
        """Show the complete memory sequence ready for the decoder."""
        title = styled_text(
            "Complete Memory Sequence",
            font_size=32, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Create final memory visualization - centered
        memory_seq = VGroup()
        for t in range(3):
            timestep_group = VGroup()
            # Show 6 image tokens per timestep
            for i in range(6):
                cell = Square(
                    side_length=0.22,
                    color=TransformerColors.token_fill,
                    fill_opacity=0.5,
                    stroke_width=1
                )
                timestep_group.add(cell)
            # Dots
            dots = styled_text("...", font_size=10, color=WHITE)
            timestep_group.add(dots)
            # Action token
            action = Square(
                side_length=0.22,
                color=TransformerColors.action_fill,
                fill_opacity=0.6,
                stroke_width=1.5
            )
            timestep_group.add(action)
            timestep_group.arrange(RIGHT, buff=0.03)
            memory_seq.add(timestep_group)

        memory_seq.arrange(RIGHT, buff=0.25)
        memory_seq.move_to(UP * 0.5)  # Center vertically, leave room for text below

        # Add timestep brackets
        brackets = VGroup()
        for i, ts in enumerate(memory_seq):
            bracket = Brace(ts, DOWN, color=GREY_B)
            label = styled_text(f"t={i}", font_size=11, color=GREY_B)
            label.next_to(bracket, DOWN, buff=0.05)
            brackets.add(VGroup(bracket, label))

        self.play(FadeIn(memory_seq, shift=UP))
        self.play(FadeIn(brackets))

        # Ready for decoder text - positioned below the memory, not to the right
        ready_text = styled_text(
            "Ready for Transformer Decoder",
            font_size=20, color=GREEN_C
        )
        ready_arrow = styled_text("→", font_size=24, color=GREEN_C)
        ready_group = VGroup(ready_arrow, ready_text).arrange(RIGHT, buff=0.2)
        ready_group.next_to(brackets, DOWN, buff=0.4)

        self.play(FadeIn(ready_group, shift=UP))

        # Final summary box - on the left side
        summary = VGroup(
            styled_text("Memory contains:", font_size=15, color=WHITE),
            styled_text("• Token + Grid position embeddings", font_size=13, color=GREY_B),
            styled_text("• Action embeddings", font_size=13, color=GREY_B),
            styled_text("• Temporal positional encoding", font_size=13, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        summary.to_edge(DOWN, buff=0.5).to_edge(LEFT, buff=1)

        self.play(FadeIn(summary, shift=UP))

        self.wait(3)

        # Final fade out
        self.play(FadeOut(*self.mobjects))


# =============================================================================
# Scene 2: Parallel Prediction
# =============================================================================

class TransformerWM_ParallelPrediction(Scene):
    """
    Visualizes how query tokens attend to the memory sequence through
    self-attention and cross-attention in the Transformer decoder.
    """

    def construct(self):
        self.camera.background_color = "#0f1419"

        self.show_title()
        self.show_query_tokens()
        self.show_query_positioning()
        self.show_decoder_overview()
        self.show_self_attention()
        self.show_cross_attention()
        self.show_layer_stacking()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "Transformer World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 2: Parallel Prediction",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_query_tokens(self):
        """Introduce learnable query tokens."""
        section_title = Text(
            "Step 1: Learnable Query Tokens",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create query token visualization
        query_tokens = VGroup()

        # 16 image query tokens
        for i in range(16):
            q = Square(
                side_length=0.4,
                color=TransformerColors.query_fill,
                fill_opacity=0.5,
                stroke_width=2
            )
            label = Text(f"Q{i + 1}", font_size=10, color=WHITE)
            label.move_to(q.get_center())
            query_tokens.add(VGroup(q, label))

        # 1 global query token
        global_q = Square(
            side_length=0.4,
            color=TransformerColors.global_token,
            fill_opacity=0.6,
            stroke_width=2
        )
        global_label = Text("G", font_size=12, color=WHITE, weight="BOLD")
        global_label.move_to(global_q.get_center())
        query_tokens.add(VGroup(global_q, global_label))

        query_tokens.arrange(RIGHT, buff=0.08)
        query_tokens.move_to(ORIGIN)

        # Braces and labels
        img_brace = Brace(query_tokens[:-1], DOWN, color=PURPLE_A)
        img_label = Text("16 Image Queries", font_size=14, color=PURPLE_A)
        img_label.next_to(img_brace, DOWN, buff=0.1)

        global_brace = Brace(query_tokens[-1], DOWN, color=ORANGE)
        global_label2 = Text("Global", font_size=14, color=ORANGE)
        global_label2.next_to(global_brace, DOWN, buff=0.1)

        self.play(LaggedStart(
            *[FadeIn(q, shift=UP) for q in query_tokens],
            lag_ratio=0.05
        ))
        self.play(FadeIn(img_brace, img_label, global_brace, global_label2))

        # Explanation
        explanation = VGroup(
            Text("• Image queries predict next VQ-VAE tokens", font_size=16, color=GREY_B),
            Text("• Global query predicts reward and done", font_size=16, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        explanation.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(explanation, shift=UP))

        self.wait(2)

        # Clean up but keep queries
        self.play(FadeOut(
            section_title, img_brace, img_label, global_brace, global_label2, explanation
        ))
        self.play(query_tokens.animate.move_to(UP * 2.5).scale(0.8))

        self.query_tokens = query_tokens

    def show_query_positioning(self):
        """Show positional encoding being added to queries."""
        section_title = styled_text(
            "Step 2: Query Position Encoding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move queries to center and add label
        self.play(self.query_tokens.animate.move_to(UP * 1.5))

        query_label = styled_text("Query Tokens", font_size=14, color=WHITE)
        query_label.next_to(self.query_tokens, UP, buff=0.2)
        self.play(FadeIn(query_label))

        # Show grid positional encoding BELOW the image queries (not overlapping)
        grid_pos = VGroup()
        for i in range(16):
            indicator = Square(
                side_length=0.25, color=YELLOW, fill_opacity=0.4, stroke_width=1
            )
            grid_pos.add(indicator)
        grid_pos.arrange(RIGHT, buff=0.06)
        grid_pos.next_to(self.query_tokens[:-1], DOWN, buff=0.4)

        grid_pos_label = styled_text("+ Grid Position (spatial)", font_size=12, color=YELLOW)
        grid_pos_label.next_to(grid_pos, DOWN, buff=0.15)

        # Arrow showing addition
        grid_arrow = Arrow(
            grid_pos.get_top(), self.query_tokens[7].get_bottom(),
            buff=0.05, color=YELLOW, stroke_width=2, tip_length=0.15
        )

        self.play(FadeIn(grid_pos, grid_pos_label, shift=UP))
        self.play(GrowArrow(grid_arrow))
        self.play(
            Indicate(self.query_tokens[:-1], color=YELLOW, scale_factor=1.05),
            run_time=0.8
        )

        # Show temporal positional encoding below grid pos
        temp_pos = VGroup()
        for i in range(17):
            indicator = Circle(
                radius=0.1, color=ORANGE, fill_opacity=0.4, stroke_width=1
            )
            temp_pos.add(indicator)
        temp_pos.arrange(RIGHT, buff=0.15)
        temp_pos.next_to(grid_pos_label, DOWN, buff=0.5)

        temp_pos_label = styled_text("+ Temporal Position (time)", font_size=12, color=ORANGE)
        temp_pos_label.next_to(temp_pos, DOWN, buff=0.1)

        temp_arrow = Arrow(
            temp_pos.get_top(), grid_pos.get_bottom() + DOWN * 0.1,
            buff=0.1, color=ORANGE, stroke_width=2, tip_length=0.15
        )

        self.play(FadeIn(temp_pos, temp_pos_label, shift=UP))
        self.play(GrowArrow(temp_arrow))

        # Merge animation
        self.play(
            FadeOut(grid_pos, grid_pos_label, temp_pos, temp_pos_label, grid_arrow, temp_arrow),
            Flash(self.query_tokens, color=YELLOW, flash_radius=0.4),
            run_time=1
        )

        caption = styled_text(
            "Queries now carry both spatial and temporal position information",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP))

        self.wait(1.5)

        self.play(FadeOut(section_title, query_label, caption))

    def show_decoder_overview(self):
        """Show transformer decoder architecture overview."""
        section_title = styled_text(
            "Step 3: Transformer Decoder",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move queries to bottom-left as input
        self.play(self.query_tokens.animate.move_to(LEFT * 4.5 + DOWN * 2).scale(0.6))

        query_label = styled_text("Input Queries", font_size=12, color=WHITE)
        query_label.next_to(self.query_tokens, DOWN, buff=0.15)
        self.play(FadeIn(query_label))

        # Create decoder layer stack - centered and cleaner
        decoder_layers = VGroup()
        layer_components = ["Self-Attn", "Cross-Attn", "FFN"]
        component_colors = [YELLOW, ORANGE, PURPLE]

        for i in range(3):
            layer = VGroup()
            for j, (comp, color) in enumerate(zip(layer_components, component_colors)):
                comp_block = RoundedRectangle(
                    width=1.6, height=0.55,
                    corner_radius=0.08, color=color,
                    fill_opacity=0.3, stroke_width=1.5
                )
                comp_text = styled_text(comp, font_size=12, color=WHITE)
                comp_text.move_to(comp_block.get_center())
                layer.add(VGroup(comp_block, comp_text))
            layer.arrange(RIGHT, buff=0.12)

            layer_frame = SurroundingRectangle(
                layer, color=GREY_D, buff=0.12,
                corner_radius=0.1, stroke_width=1
            )
            layer_label = styled_text(f"Layer {i + 1}", font_size=11, color=GREY_B)
            layer_label.next_to(layer_frame, RIGHT, buff=0.15)

            decoder_layers.add(VGroup(layer_frame, layer, layer_label))

        decoder_layers.arrange(UP, buff=0.2)
        decoder_layers.move_to(ORIGIN)

        # Input arrow from queries to decoder (to center of first layer)
        input_arrow = Arrow(
            self.query_tokens.get_top(),
            decoder_layers[0].get_bottom() + DOWN * 0.05,
            buff=0.1, color=WHITE, stroke_width=3, tip_length=0.15
        )

        self.play(GrowArrow(input_arrow))
        self.play(LaggedStart(
            *[FadeIn(layer, shift=UP) for layer in decoder_layers],
            lag_ratio=0.15
        ))

        # Output queries (similar style to input)
        output_queries = VGroup()
        for i in range(17):
            color = GREEN_C if i < 16 else ORANGE
            q = Square(side_length=0.18, color=color, fill_opacity=0.6, stroke_width=1.5)
            output_queries.add(q)
        output_queries.arrange(RIGHT, buff=0.03)
        output_queries.next_to(decoder_layers[-1], UP, buff=0.5)

        output_label = styled_text("Refined Queries", font_size=12, color=GREEN_C)
        output_label.next_to(output_queries, UP, buff=0.1)

        output_arrow = Arrow(
            decoder_layers[-1].get_top(),
            output_queries.get_bottom(),
            buff=0.08, color=GREEN_C, stroke_width=3, tip_length=0.15
        )

        self.play(GrowArrow(output_arrow))
        self.play(FadeIn(output_queries, output_label, shift=UP))

        # Note about 10 layers
        note = styled_text("(Full model: 10 layers)", font_size=12, color=GREY_B)
        note.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note))

        self.wait(1.5)

        # Store and clean up for next section
        self.decoder_layers = decoder_layers

        self.play(FadeOut(
            section_title, query_label, input_arrow, output_arrow, output_label, output_queries, note
        ))

    def show_self_attention(self):
        """Visualize self-attention among query tokens."""
        section_title = styled_text(
            "Step 4: Self-Attention",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Highlight self-attention block
        self_attn_block = self.decoder_layers[1][1][0]  # Middle layer, self-attn
        self.play(Indicate(self_attn_block, color=YELLOW, scale_factor=1.2))

        # Create focused view of self-attention
        self.play(
            self.query_tokens.animate.move_to(LEFT * 3 + DOWN * 0.3).scale(1.2),
            FadeOut(self.decoder_layers)
        )

        # Create attention visualization
        attn_label = styled_text("Queries attend to each other", font_size=18, color=WHITE)
        attn_label.to_edge(UP, buff=1.2)
        self.play(FadeIn(attn_label))

        # Show attention arrows between a few queries - smaller tip_length
        sample_arrows = VGroup()
        connections = [(0, 3), (2, 5), (4, 8), (7, 10), (12, 15)]

        for src_idx, tgt_idx in connections:
            src = self.query_tokens[src_idx]
            tgt = self.query_tokens[tgt_idx]
            arrow = CurvedArrow(
                src.get_center() + DOWN * 0.12,
                tgt.get_center() + DOWN * 0.12,
                angle=-0.4, color=YELLOW,
                stroke_width=1.5, stroke_opacity=0.7,
                tip_length=0.1  # Smaller tip
            )
            sample_arrows.add(arrow)

        self.play(LaggedStart(
            *[Create(arr) for arr in sample_arrows],
            lag_ratio=0.1
        ), run_time=1.2)

        # Show attention matrix with variable weights
        attn_matrix = make_attention_matrix(
            rows=8, cols=8, cell_size=0.22,
            mask_pattern="full_heatmap", block_size=8
        )
        attn_matrix.move_to(RIGHT * 3.5 + DOWN * 0.3)

        matrix_label = styled_text("Self-Attention Weights", font_size=12, color=GREY_B)
        matrix_label.next_to(attn_matrix, UP, buff=0.15)

        self.play(FadeIn(attn_matrix, matrix_label, shift=LEFT))

        # Explanation
        explanation = styled_text(
            "All queries can see all other queries (during inference)",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation, shift=UP))

        self.wait(2)

        self.play(FadeOut(
            section_title, attn_label, sample_arrows, attn_matrix, matrix_label, explanation
        ))

    def show_cross_attention(self):
        """Visualize cross-attention from queries to memory."""
        section_title = styled_text(
            "Step 5: Cross-Attention",
            font_size=28, color=ORANGE
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move queries up
        self.play(self.query_tokens.animate.move_to(UP * 1.6).scale(0.75))

        query_label = styled_text("Queries (Q)", font_size=13, color=PURPLE_A)
        query_label.next_to(self.query_tokens, LEFT, buff=0.2)
        self.play(FadeIn(query_label))

        # Create memory sequence
        memory_seq = VGroup()
        for t in range(3):
            timestep = VGroup()
            for i in range(5):
                color = TransformerColors.token_fill if i < 4 else TransformerColors.action_fill
                cell = Square(
                    side_length=0.22,
                    color=color,
                    fill_opacity=0.5 if i < 4 else 0.6,
                    stroke_width=1
                )
                timestep.add(cell)
            dots = styled_text("...", font_size=9, color=WHITE)
            timestep.add(dots)
            timestep.arrange(RIGHT, buff=0.03)
            memory_seq.add(timestep)

        memory_seq.arrange(RIGHT, buff=0.15)
        memory_seq.move_to(DOWN * 1)

        # Add timestep labels
        mem_labels = VGroup()
        for i, ts in enumerate(memory_seq):
            label = styled_text(f"t={i}", font_size=10, color=GREY_B)
            label.next_to(ts, DOWN, buff=0.08)
            mem_labels.add(label)

        memory_label = styled_text("Memory (K, V)", font_size=13, color=BLUE_C)
        memory_label.next_to(memory_seq, LEFT, buff=0.2)

        self.play(
            FadeIn(memory_seq, shift=UP),
            FadeIn(mem_labels),
            FadeIn(memory_label)
        )

        # Show attention arrows from queries to memory - smaller tips, fewer arrows
        cross_arrows = VGroup()
        for q_idx in [2, 8, 14]:  # Sample queries
            q = self.query_tokens[q_idx]
            for t_idx, ts in enumerate(memory_seq):
                target_cell = ts[2]
                # Vary opacity based on "attention weight"
                opacity = 0.3 + 0.4 * np.random.random()
                arrow = CurvedArrow(
                    q.get_bottom() + DOWN * 0.02,
                    target_cell.get_top() + UP * 0.02,
                    angle=0.25,
                    color=ORANGE,
                    stroke_width=1.2,
                    stroke_opacity=opacity,
                    tip_length=0.08  # Smaller tip
                )
                cross_arrows.add(arrow)

        self.play(LaggedStart(
            *[Create(arr) for arr in cross_arrows],
            lag_ratio=0.02
        ), run_time=1.5)

        # Show attention matrix with variable weights
        cross_matrix = make_attention_matrix(
            rows=6, cols=9, cell_size=0.2,
            mask_pattern="full_heatmap", block_size=3
        )
        cross_matrix.move_to(RIGHT * 4.5 + DOWN * 0.3)

        matrix_label = styled_text("Cross-Attention Weights", font_size=11, color=GREY_B)
        matrix_label.next_to(cross_matrix, UP, buff=0.12)

        self.play(FadeIn(cross_matrix, matrix_label, shift=LEFT))

        # Explanation
        explanation = styled_text(
            "Queries attend to the entire memory history",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(explanation, shift=UP))

        self.wait(2)

        # Store memory for later
        self.memory_seq = memory_seq
        self.mem_labels = mem_labels

        self.play(FadeOut(
            section_title, query_label, memory_label, cross_arrows,
            cross_matrix, matrix_label, explanation
        ))

    def show_layer_stacking(self):
        """Show information flowing through multiple decoder layers."""
        section_title = styled_text(
            "Step 6: Layer-by-Layer Refinement",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Clean up previous elements
        self.play(FadeOut(self.memory_seq, self.mem_labels, self.query_tokens))

        # Create input queries strip (same style as output)
        input_queries = VGroup()
        for i in range(17):
            color = PURPLE_A if i < 16 else ORANGE
            q = Square(side_length=0.18, color=color, fill_opacity=0.5, stroke_width=1.5)
            input_queries.add(q)
        input_queries.arrange(RIGHT, buff=0.03)
        input_queries.move_to(DOWN * 2.8)

        input_label = styled_text("Input Queries", font_size=12, color=GREY_B)
        input_label.next_to(input_queries, DOWN, buff=0.12)

        self.play(FadeIn(input_queries, input_label, shift=UP))

        # Create vertical layer stack - centered
        layers = VGroup()
        layer_height = 0.55
        layer_width = 4.5

        for i in range(4):
            layer = RoundedRectangle(
                width=layer_width, height=layer_height,
                corner_radius=0.1, color=PURPLE,
                fill_opacity=0.15 + i * 0.1,
                stroke_width=1.5
            )
            label = styled_text(f"Decoder Layer {i + 1}", font_size=13, color=WHITE)
            label.move_to(layer.get_center())
            layers.add(VGroup(layer, label))

        layers.arrange(UP, buff=0.2)
        layers.move_to(UP * 0.2)

        # Output queries strip
        output_queries = VGroup()
        for i in range(17):
            color = GREEN_C if i < 16 else ORANGE
            q = Square(side_length=0.18, color=color, fill_opacity=0.6, stroke_width=1.5)
            output_queries.add(q)
        output_queries.arrange(RIGHT, buff=0.03)
        output_queries.next_to(layers[-1], UP, buff=0.35)

        output_label = styled_text("Refined Queries", font_size=12, color=GREEN_C)
        output_label.next_to(output_queries, UP, buff=0.1)

        # Arrows - input arrow from CENTER of input queries to CENTER-BOTTOM of first layer
        input_arrow = Arrow(
            input_queries.get_top(),
            layers[0].get_bottom(),
            buff=0.08, color=WHITE, stroke_width=3, tip_length=0.15
        )

        between_arrows = VGroup()
        for i in range(len(layers) - 1):
            arr = Arrow(
                layers[i].get_top(), layers[i + 1].get_bottom(),
                buff=0.04, color=GREY_B, stroke_width=2, tip_length=0.12
            )
            between_arrows.add(arr)

        output_arrow = Arrow(
            layers[-1].get_top(), output_queries.get_bottom(),
            buff=0.08, color=GREEN_C, stroke_width=3, tip_length=0.15
        )

        # Animate
        self.play(GrowArrow(input_arrow))

        for layer in layers:
            self.play(FadeIn(layer, shift=UP), run_time=0.35)

        self.play(LaggedStart(*[GrowArrow(arr) for arr in between_arrows], lag_ratio=0.15))
        self.play(GrowArrow(output_arrow))
        self.play(FadeIn(output_queries, output_label, shift=UP))

        # Note about 10 layers
        note = styled_text("(10 layers total)", font_size=11, color=GREY_B)
        note.next_to(layers, RIGHT, buff=0.4)
        self.play(FadeIn(note))

        # Final message
        final_msg = styled_text(
            "Each layer refines the query representations",
            font_size=17, color=GREY_B
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(final_msg, shift=UP))

        self.wait(3)

        # Final fade out
        self.play(FadeOut(*self.mobjects))


# =============================================================================
# Scene 3: Prediction Heads
# =============================================================================

class TransformerWM_PredictionHeads(Scene):
    """
    Visualizes how the refined query outputs are used by prediction heads
    to generate next latent tokens, reward, and done signals.
    """

    def construct(self):
        self.camera.background_color = "#0f1419"

        self.show_title()
        self.show_decoder_output()
        self.show_latent_prediction()
        self.show_global_predictions()
        self.show_reconstruction()
        self.show_summary()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "Transformer World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 3: Prediction Heads",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_decoder_output(self):
        """Show the output from the decoder."""
        section_title = styled_text(
            "Decoder Output: Refined Query Embeddings",
            font_size=26, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create 16 image query outputs as a 4x4 grid
        img_outputs = VGroup()
        for i in range(16):
            bar = make_embedding_bar(width=0.85, height=0.22, gradient=True)
            img_outputs.add(bar)
        img_outputs.arrange_in_grid(rows=4, cols=4, buff=0.1)
        img_outputs.move_to(UP * 0.5)

        # 1 global query output below the grid
        global_output = RoundedRectangle(
            width=1.5, height=0.3,
            corner_radius=0.08, color=ORANGE,
            fill_opacity=0.5, stroke_width=2
        )
        global_output.next_to(img_outputs, DOWN, buff=0.5)

        # Labels
        img_label = styled_text("16 Image Query Outputs\n(4×4 grid)", font_size=13, color=PURPLE_A)
        img_label.next_to(img_outputs, LEFT, buff=0.4)

        global_label = styled_text("Global Query Output", font_size=13, color=ORANGE)
        global_label.next_to(global_output, LEFT, buff=0.4)

        # Animate - first 4x4 grid, then global
        self.play(LaggedStart(
            *[FadeIn(bar, shift=UP) for bar in img_outputs],
            lag_ratio=0.03
        ))
        self.play(FadeIn(img_label))

        self.play(FadeIn(global_output, shift=UP))
        self.play(FadeIn(global_label))

        # Dimension annotation
        dim_text = styled_text("Each embedding: 512-dimensional", font_size=14, color=GREY_B)
        dim_text.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(dim_text, shift=UP))

        self.wait(1.5)

        # Store for later (combine into query_outputs for compatibility)
        self.query_outputs = VGroup(*img_outputs, global_output)
        self.img_outputs = img_outputs
        self.global_output = global_output

        self.play(FadeOut(
            section_title, img_label, global_label, dim_text
        ))

    def show_latent_prediction(self):
        """Show next latent tokens prediction."""
        section_title = Text(
            "Step 1: Next Latent Tokens Prediction",
            font_size=26, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Reorganize - move image outputs to left
        self.play(
            self.img_outputs.animate.arrange_in_grid(rows=4, cols=4, buff=0.1).move_to(LEFT * 4 + UP * 0.5).scale(0.8),
            self.global_output.animate.move_to(LEFT * 4 + DOWN * 2).scale(0.6).set_opacity(0.3)
        )

        img_label = Text("Image Query\nOutputs", font_size=12, color=WHITE)
        img_label.next_to(self.img_outputs, UP, buff=0.15)
        self.play(FadeIn(img_label))

        # Prediction head
        head_block = make_prediction_head(
            width=2.2, height=0.8,
            label="Next Latent Head\n(Linear → 512)", color=PURPLE
        )
        head_block.move_to(LEFT * 0.5 + UP * 0.5)

        arrow1 = Arrow(
            self.img_outputs.get_right(),
            head_block.get_left(),
            buff=0.1, color=WHITE
        )

        self.play(GrowArrow(arrow1), FadeIn(head_block, shift=RIGHT))

        # Output logits grid
        logits_grid = VGroup()
        for _ in range(16):
            cell = Square(
                side_length=0.35,
                color=GREEN_C,
                fill_opacity=0.4,
                stroke_width=1.5
            )
            logits_grid.add(cell)
        logits_grid.arrange_in_grid(rows=4, cols=4, buff=0.08)
        logits_grid.move_to(RIGHT * 3 + UP * 0.5)

        logits_label = Text("16 × 512 Logits", font_size=14, color=WHITE)
        logits_label.next_to(logits_grid, UP, buff=0.15)

        arrow2 = Arrow(
            head_block.get_right(),
            logits_grid.get_left(),
            buff=0.1, color=WHITE
        )

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(logits_grid, logits_label, shift=RIGHT))

        # Show probability distribution for one cell
        prob_chart = self._create_prob_bar_chart()
        prob_chart.move_to(RIGHT * 3 + DOWN * 2)

        prob_label = Text("Probability over\n512 codebook entries", font_size=11, color=GREY_B)
        prob_label.next_to(prob_chart, DOWN, buff=0.15)

        # Highlight one cell
        highlight = SurroundingRectangle(logits_grid[5], color=YELLOW, buff=0.02)
        self.play(Create(highlight))

        prob_arrow = Arrow(
            logits_grid[5].get_bottom(),
            prob_chart.get_top(),
            buff=0.15, color=YELLOW
        )
        self.play(GrowArrow(prob_arrow), FadeIn(prob_chart, prob_label, shift=UP))

        # Sampling
        sample_text = Text("→ argmax or sample", font_size=12, color=GREEN_C)
        sample_text.next_to(prob_chart, RIGHT, buff=0.2)
        self.play(FadeIn(sample_text, shift=LEFT))

        self.wait(2)

        # Clean up
        self.play(FadeOut(
            section_title, img_label, arrow1, head_block, arrow2,
            logits_grid, logits_label, highlight, prob_arrow, prob_chart,
            prob_label, sample_text
        ))

    def _create_prob_bar_chart(self) -> VGroup:
        """Create a small probability distribution bar chart."""
        chart = VGroup()
        probs = [0.35, 0.25, 0.15, 0.1, 0.08, 0.07]
        labels = ["127", "43", "512", "89", "...", "..."]

        max_width = 1.2
        bar_height = 0.15

        for i, (prob, label) in enumerate(zip(probs, labels)):
            bar = Rectangle(
                width=prob * max_width,
                height=bar_height,
                color=TEAL_C,
                fill_opacity=0.7,
                stroke_width=0
            )
            bar.align_to(ORIGIN, LEFT)

            lbl = Text(label, font_size=9, color=GREY_B)
            lbl.next_to(bar, LEFT, buff=0.1)

            chart.add(VGroup(lbl, bar))

        chart.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        return chart

    def show_global_predictions(self):
        """Show reward and done predictions from global token."""
        section_title = Text(
            "Step 2: Global Token Predictions",
            font_size=26, color=ORANGE
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Bring global output to focus
        self.play(
            self.global_output.animate.move_to(LEFT * 4).scale(1.5).set_opacity(1),
            self.img_outputs.animate.move_to(LEFT * 4 + UP * 2.5).scale(0.5).set_opacity(0.3)
        )

        global_label = Text("Global Query Output", font_size=14, color=ORANGE)
        global_label.next_to(self.global_output, UP, buff=0.15)
        self.play(FadeIn(global_label))

        # Two prediction heads branching out
        reward_head = make_prediction_head(
            width=1.8, height=0.6,
            label="Reward Head", color=GREEN_C
        )
        reward_head.move_to(RIGHT * 1 + UP * 1.2)

        done_head = make_prediction_head(
            width=1.8, height=0.6,
            label="Done Head", color=RED_C
        )
        done_head.move_to(RIGHT * 1 + DOWN * 1.2)

        # Arrows
        arrow_reward = Arrow(
            self.global_output.get_right(),
            reward_head.get_left(),
            buff=0.1, color=GREEN_C
        )
        arrow_done = Arrow(
            self.global_output.get_right(),
            done_head.get_left(),
            buff=0.1, color=RED_C
        )

        self.play(
            GrowArrow(arrow_reward),
            GrowArrow(arrow_done),
            FadeIn(reward_head, done_head, shift=RIGHT)
        )

        # Output values
        reward_value = VGroup(
            Text("Reward:", font_size=14, color=WHITE),
            Text("+0.82", font_size=18, color=GREEN_C, weight="BOLD")
        ).arrange(RIGHT, buff=0.2)
        reward_value.next_to(reward_head, RIGHT, buff=0.3)

        done_value = VGroup(
            Text("Done:", font_size=14, color=WHITE),
            Text("0.03", font_size=18, color=RED_C, weight="BOLD")
        ).arrange(RIGHT, buff=0.2)
        done_value.next_to(done_head, RIGHT, buff=0.3)

        self.play(FadeIn(reward_value, done_value, shift=LEFT))

        # Explanation
        explanation = VGroup(
            Text("• Reward: scalar value for RL training", font_size=14, color=GREY_B),
            Text("• Done: probability of episode termination", font_size=14, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        explanation.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation, shift=UP))

        self.wait(2)

        self.play(FadeOut(
            section_title, global_label, reward_head, done_head,
            arrow_reward, arrow_done, reward_value, done_value, explanation
        ))

    def show_reconstruction(self):
        """Show VQ-VAE decoding back to image."""
        section_title = Text(
            "Step 3: Reconstruct Predicted Frame",
            font_size=26, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Clean up previous
        self.play(FadeOut(self.img_outputs, self.global_output))

        # Predicted tokens grid
        pred_tokens = make_token_grid_indexed(
            rows=4, cols=4, cell_size=0.45,
            color=GREEN_C, show_indices=False
        )
        pred_tokens.move_to(LEFT * 4)

        token_label = Text("Predicted Token\nIndices", font_size=14, color=WHITE)
        token_label.next_to(pred_tokens, UP, buff=0.2)

        self.play(FadeIn(pred_tokens, token_label, shift=LEFT))

        # VQ-VAE Decoder
        decoder_block = block("VQ-VAE\nDecoder", width=2.2, height=1.2, kind="encode", font_size=18)
        decoder_block.move_to(ORIGIN)

        arrow1 = Arrow(pred_tokens.get_right(), decoder_block.get_left(), buff=0.2, color=WHITE)

        self.play(GrowArrow(arrow1), FadeIn(decoder_block, shift=RIGHT))

        # Reconstructed frame
        recon_frame = make_frame_placeholder(width=2.0, height=2.0)
        recon_frame.move_to(RIGHT * 4)

        frame_label = Text("Predicted Next\nObservation", font_size=14, color=GREEN_C)
        frame_label.next_to(recon_frame, UP, buff=0.2)

        arrow2 = Arrow(decoder_block.get_right(), recon_frame.get_left(), buff=0.2, color=WHITE)

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(recon_frame, frame_label, shift=RIGHT))

        # Highlight that this is the prediction
        highlight = SurroundingRectangle(recon_frame, color=GREEN_C, buff=0.1)
        self.play(Create(highlight))

        self.wait(2)

        self.play(FadeOut(
            section_title, pred_tokens, token_label, arrow1, decoder_block,
            arrow2, recon_frame, frame_label, highlight
        ))

    def show_summary(self):
        """Show final summary of prediction outputs."""
        title = Text(
            "World Model Predictions Summary",
            font_size=32, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Create summary diagram
        summary = VGroup()

        # Input
        input_box = RoundedRectangle(
            width=2.5, height=1,
            corner_radius=0.1, color=BLUE_C,
            fill_opacity=0.2, stroke_width=2
        )
        input_text = Text("History\n(obs + actions)", font_size=14, color=WHITE)
        input_text.move_to(input_box.get_center())
        input_group = VGroup(input_box, input_text)

        # Model
        model_box = RoundedRectangle(
            width=3, height=1.5,
            corner_radius=0.1, color=PURPLE,
            fill_opacity=0.2, stroke_width=2
        )
        model_text = Text("Transformer\nWorld Model", font_size=16, color=WHITE, weight="BOLD")
        model_text.move_to(model_box.get_center())
        model_group = VGroup(model_box, model_text)

        # Outputs
        output1 = VGroup(
            Rectangle(width=1.8, height=0.5, color=GREEN_C, fill_opacity=0.3, stroke_width=1.5),
            Text("Next Observation", font_size=11, color=WHITE)
        )
        output1[1].move_to(output1[0].get_center())

        output2 = VGroup(
            Rectangle(width=1.8, height=0.5, color=TEAL_C, fill_opacity=0.3, stroke_width=1.5),
            Text("Reward", font_size=11, color=WHITE)
        )
        output2[1].move_to(output2[0].get_center())

        output3 = VGroup(
            Rectangle(width=1.8, height=0.5, color=RED_C, fill_opacity=0.3, stroke_width=1.5),
            Text("Done Flag", font_size=11, color=WHITE)
        )
        output3[1].move_to(output3[0].get_center())

        outputs = VGroup(output1, output2, output3)
        outputs.arrange(DOWN, buff=0.2)

        # Position
        input_group.move_to(LEFT * 4)
        model_group.move_to(ORIGIN)
        outputs.move_to(RIGHT * 4)

        # Arrows
        arr1 = Arrow(input_group.get_right(), model_group.get_left(), buff=0.1, color=WHITE)
        arr2 = Arrow(model_group.get_right(), outputs.get_left(), buff=0.1, color=WHITE)

        summary.add(input_group, arr1, model_group, arr2, outputs)

        self.play(
            FadeIn(input_group, shift=RIGHT),
            GrowArrow(arr1),
            FadeIn(model_group, shift=UP),
            GrowArrow(arr2),
            LaggedStart(*[FadeIn(o, shift=LEFT) for o in outputs], lag_ratio=0.2)
        )

        # Key insight
        insight = Text(
            "The world model learns to simulate the environment!",
            font_size=18, color=YELLOW
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(insight, shift=UP))

        self.wait(3)

        self.play(FadeOut(*self.mobjects))


# =============================================================================
# Scene 4: Teacher Forcing Masking
# =============================================================================

class TransformerWM_Masking(Scene):
    """
    Visualizes the Temporal Block Teacher Forcing (T-BTF) masking scheme
    used during training to process entire sequences in parallel.
    """

    def construct(self):
        self.camera.background_color = "#0f1419"

        self.show_title()
        self.show_training_context()
        self.show_block_diagonal_mask()
        self.show_causal_cross_mask()
        self.show_combined_view()
        self.show_why_it_works()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "Transformer World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 4: Training Masking (T-BTF)",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_training_context(self):
        """Explain the training vs inference difference."""
        section_title = Text(
            "Training vs Inference",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Two columns
        inference_title = Text("Inference", font_size=22, color=GREEN_C, weight="BOLD")
        training_title = Text("Training", font_size=22, color=ORANGE, weight="BOLD")

        inference_title.move_to(LEFT * 3 + UP * 1.5)
        training_title.move_to(RIGHT * 3 + UP * 1.5)

        self.play(FadeIn(inference_title, training_title))

        # Inference description
        inf_desc = VGroup(
            Text("• Process one step", font_size=14, color=GREY_B),
            Text("• Sequential generation", font_size=14, color=GREY_B),
            Text("• No masking needed", font_size=14, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        inf_desc.next_to(inference_title, DOWN, buff=0.4)

        # Training description
        train_desc = VGroup(
            Text("• Process H steps parallel", font_size=14, color=GREY_B),
            Text("• Teacher forcing", font_size=14, color=GREY_B),
            Text("• Requires masking!", font_size=14, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        train_desc.next_to(training_title, DOWN, buff=0.4)

        self.play(FadeIn(inf_desc, train_desc))

        # Visual: single vs parallel
        single_seq = VGroup()
        for i in range(3):
            cell = Square(side_length=0.4, color=GREEN_C, fill_opacity=0.4)
            single_seq.add(cell)
        single_seq.arrange(RIGHT, buff=0.2)
        single_seq.move_to(LEFT * 3 + DOWN * 1.5)

        single_arrows = VGroup()
        for i in range(2):
            arr = Arrow(
                single_seq[i].get_right(),
                single_seq[i + 1].get_left(),
                buff=0.05, color=WHITE, stroke_width=2
            )
            single_arrows.add(arr)

        parallel_seq = VGroup()
        for i in range(3):
            cell = Square(side_length=0.4, color=ORANGE, fill_opacity=0.4)
            parallel_seq.add(cell)
        parallel_seq.arrange(RIGHT, buff=0.2)
        parallel_seq.move_to(RIGHT * 3 + DOWN * 1.5)

        # All at once indication
        parallel_brace = Brace(parallel_seq, DOWN, color=WHITE)
        parallel_label = Text("All at once", font_size=12, color=WHITE)
        parallel_label.next_to(parallel_brace, DOWN, buff=0.1)

        self.play(
            FadeIn(single_seq),
            LaggedStart(*[GrowArrow(a) for a in single_arrows], lag_ratio=0.3),
            FadeIn(parallel_seq, parallel_brace, parallel_label)
        )

        self.wait(2)

        # Key message
        key_msg = Text(
            "Masking prevents information leakage from future timesteps",
            font_size=16, color=YELLOW
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(key_msg, shift=UP))

        self.wait(2)

        self.play(FadeOut(*self.mobjects))

    def show_block_diagonal_mask(self):
        """Visualize block-diagonal self-attention mask."""
        section_title = Text(
            "Self-Attention: Block-Diagonal Mask",
            font_size=26, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create mask matrix visualization
        num_steps = 3
        queries_per_step = 4  # Simplified from 17

        matrix = VGroup()
        cell_size = 0.35

        for r in range(num_steps * queries_per_step):
            for c in range(num_steps * queries_per_step):
                cell = Square(
                    side_length=cell_size,
                    stroke_width=0.5,
                    stroke_color=GREY_D
                )

                # Block diagonal: same timestep only
                block_r = r // queries_per_step
                block_c = c // queries_per_step

                if block_r == block_c:
                    cell.set_fill(GREEN_C, opacity=0.7)
                else:
                    cell.set_fill(RED_C, opacity=0.2)

                matrix.add(cell)

        matrix.arrange_in_grid(
            rows=num_steps * queries_per_step,
            cols=num_steps * queries_per_step,
            buff=0.02
        )
        matrix.move_to(LEFT * 1.5)

        # Add step labels
        step_labels_row = VGroup()
        step_labels_col = VGroup()
        for i in range(num_steps):
            row_label = Text(f"t={i}", font_size=12, color=GREY_B)
            col_label = Text(f"t={i}", font_size=12, color=GREY_B)
            step_labels_row.add(row_label)
            step_labels_col.add(col_label)

        # Position row labels (approximate)
        total_rows = num_steps * queries_per_step
        for i, label in enumerate(step_labels_row):
            row_idx = i * queries_per_step + queries_per_step // 2
            label.next_to(matrix, LEFT, buff=0.3)
            label.shift(UP * (total_rows / 2 - row_idx - 0.5) * (cell_size + 0.02))

        # Position col labels
        for i, label in enumerate(step_labels_col):
            col_idx = i * queries_per_step + queries_per_step // 2
            label.next_to(matrix, UP, buff=0.2)
            label.shift(RIGHT * (col_idx - total_rows / 2 + 0.5) * (cell_size + 0.02))

        self.play(FadeIn(matrix, shift=UP))
        self.play(FadeIn(step_labels_row, step_labels_col))

        # Legend
        legend = VGroup(
            VGroup(
                Square(side_length=0.25, color=GREEN_C, fill_opacity=0.7),
                Text("Can attend", font_size=12, color=WHITE)
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Square(side_length=0.25, color=RED_C, fill_opacity=0.2),
                Text("Blocked", font_size=12, color=WHITE)
            ).arrange(RIGHT, buff=0.1)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.move_to(RIGHT * 4 + UP * 1)

        self.play(FadeIn(legend))

        # Explanation
        explanation = VGroup(
            Text("Queries at timestep t can only", font_size=16, color=WHITE),
            Text("attend to queries at timestep t", font_size=16, color=YELLOW),
        ).arrange(DOWN, buff=0.1)
        explanation.move_to(RIGHT * 4 + DOWN * 0.5)

        self.play(FadeIn(explanation))

        # Animate highlighting blocks
        for i in range(num_steps):
            block_cells = VGroup()
            for r in range(queries_per_step):
                for c in range(queries_per_step):
                    idx = (i * queries_per_step + r) * (num_steps * queries_per_step) + (i * queries_per_step + c)
                    block_cells.add(matrix[idx])

            self.play(
                Indicate(block_cells, color=YELLOW, scale_factor=1.05),
                run_time=0.5
            )

        self.wait(2)

        # Store and clean up
        self.self_attn_matrix = matrix

        self.play(FadeOut(
            section_title, step_labels_row, step_labels_col, legend, explanation
        ))

    def show_causal_cross_mask(self):
        """Visualize causal cross-attention mask."""
        section_title = Text(
            "Cross-Attention: Causal Mask",
            font_size=26, color=ORANGE
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move self-attention matrix to left
        self.play(self.self_attn_matrix.animate.move_to(LEFT * 4.5).scale(0.6))

        sa_label = Text("Self-Attention", font_size=12, color=YELLOW)
        sa_label.next_to(self.self_attn_matrix, UP, buff=0.15)
        self.play(FadeIn(sa_label))

        # Create causal cross-attention mask
        num_steps = 3
        queries_per_step = 4
        mem_per_step = 5  # tokens + action

        cross_matrix = VGroup()
        cell_size = 0.28

        for r in range(num_steps * queries_per_step):
            for c in range(num_steps * mem_per_step):
                cell = Square(
                    side_length=cell_size,
                    stroke_width=0.5,
                    stroke_color=GREY_D
                )

                # Causal: queries at t can see memory up to and including t
                query_step = r // queries_per_step
                mem_step = c // mem_per_step

                if mem_step <= query_step:
                    cell.set_fill(GREEN_C, opacity=0.7)
                else:
                    cell.set_fill(RED_C, opacity=0.2)

                cross_matrix.add(cell)

        cross_matrix.arrange_in_grid(
            rows=num_steps * queries_per_step,
            cols=num_steps * mem_per_step,
            buff=0.02
        )
        cross_matrix.move_to(RIGHT * 1.5)

        # Labels
        query_label = Text("Queries", font_size=11, color=PURPLE_A)
        query_label.next_to(cross_matrix, LEFT, buff=0.2)

        mem_label = Text("Memory (t=0,1,2)", font_size=11, color=BLUE_C)
        mem_label.next_to(cross_matrix, UP, buff=0.15)

        self.play(FadeIn(cross_matrix, query_label, mem_label, shift=RIGHT))

        ca_label = Text("Cross-Attention", font_size=12, color=ORANGE)
        ca_label.next_to(cross_matrix, DOWN, buff=0.3)
        self.play(FadeIn(ca_label))

        # Explanation
        explanation = VGroup(
            Text("Queries at timestep t can only", font_size=14, color=WHITE),
            Text("attend to memory ≤ t", font_size=14, color=ORANGE),
        ).arrange(DOWN, buff=0.1)
        explanation.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(explanation))

        # Animate the causal structure
        for i in range(num_steps):
            # Highlight queries at step i
            row_start = i * queries_per_step
            row_end = (i + 1) * queries_per_step
            visible_cols = (i + 1) * mem_per_step

            highlight_cells = VGroup()
            for r in range(row_start, row_end):
                for c in range(visible_cols):
                    idx = r * (num_steps * mem_per_step) + c
                    highlight_cells.add(cross_matrix[idx])

            self.play(
                Indicate(highlight_cells, color=ORANGE, scale_factor=1.03),
                run_time=0.6
            )

        self.wait(2)

        self.play(FadeOut(*self.mobjects))

    def show_combined_view(self):
        """Show both masks side by side with context."""
        section_title = styled_text(
            "Combined Masking Strategy",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Self-attention mask: 9x9 (3 timesteps × 3 queries per step)
        sa_mask = make_attention_matrix(
            rows=9, cols=9, cell_size=0.26,
            mask_pattern="block_diagonal", block_size=3
        )
        sa_mask.move_to(LEFT * 3.2 + DOWN * 0.2)

        sa_title = styled_text("Self-Attention\n(Block Diagonal)", font_size=13, color=YELLOW)
        sa_title.next_to(sa_mask, UP, buff=0.15)

        # Cross-attention mask: 9 query rows × 9 memory cols (3 timesteps × 3 tokens)
        # This ensures consistent 3 timesteps for both queries and memory
        ca_mask = make_attention_matrix(
            rows=9, cols=9, cell_size=0.26,
            mask_pattern="causal", block_size=3
        )
        ca_mask.move_to(RIGHT * 2.8 + DOWN * 0.2)

        ca_title = styled_text("Cross-Attention\n(Causal)", font_size=13, color=ORANGE)
        ca_title.next_to(ca_mask, UP, buff=0.15)

        self.play(
            FadeIn(sa_mask, sa_title, shift=UP),
            FadeIn(ca_mask, ca_title, shift=UP)
        )

        # Arrow showing flow
        flow_arrow = Arrow(
            sa_mask.get_right(), ca_mask.get_left(),
            buff=0.25, color=WHITE, tip_length=0.15
        )
        flow_text = styled_text("then", font_size=13, color=WHITE)
        flow_text.next_to(flow_arrow, UP, buff=0.08)

        self.play(GrowArrow(flow_arrow), FadeIn(flow_text))

        # Summary points
        summary = VGroup(
            styled_text("✓ Self-attn: queries only see same-timestep queries", font_size=12, color=YELLOW),
            styled_text("✓ Cross-attn: queries only see past/current memory", font_size=12, color=ORANGE),
            styled_text("= No information leakage from the future!", font_size=13, color=GREEN_C, weight="BOLD"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        summary.to_edge(DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(s, shift=LEFT) for s in summary], lag_ratio=0.3))

        self.wait(3)

        self.play(FadeOut(*self.mobjects))

    def show_why_it_works(self):
        """Explain the benefits of T-BTF."""
        title = Text(
            "Why Temporal Block Teacher Forcing?",
            font_size=28, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        benefits = VGroup(
            VGroup(
                Text("1. Efficiency", font_size=20, color=YELLOW, weight="BOLD"),
                Text("   Train on entire sequence in one forward pass", font_size=16, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),

            VGroup(
                Text("2. Parallelism", font_size=20, color=YELLOW, weight="BOLD"),
                Text("   GPU can process all timesteps simultaneously", font_size=16, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),

            VGroup(
                Text("3. Causality", font_size=20, color=YELLOW, weight="BOLD"),
                Text("   Masking ensures model can't cheat by looking ahead", font_size=16, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),

            VGroup(
                Text("4. Consistency", font_size=20, color=YELLOW, weight="BOLD"),
                Text("   Training behavior matches inference behavior", font_size=16, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1),
        )

        benefits.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        benefits.move_to(ORIGIN)

        self.play(LaggedStart(
            *[FadeIn(b, shift=LEFT) for b in benefits],
            lag_ratio=0.3
        ))

        self.wait(3)

        # Final message
        final = Text(
            "→ This enables fast and stable world model training!",
            font_size=18, color=GREEN_C
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(final, shift=UP))

        self.wait(3)

        self.play(FadeOut(*self.mobjects))


# =============================================================================
# Combined Scene: Full Transformer World Model
# =============================================================================

class TransformerWM_Overview(Scene):
    """
    Complete scene that combines all parts of the Transformer World Model
    visualization in sequence. Run individual scenes for testing:
    - TransformerWM_MemoryConstruction
    - TransformerWM_ParallelPrediction
    - TransformerWM_PredictionHeads
    - TransformerWM_Masking
    """

    def construct(self):
        self.camera.background_color = "#0f1419"

        # Introduction
        self.show_intro()

        # Note: In a full implementation, you would chain the scenes
        # For now, this shows the intro/outro structure
        self.show_overview()
        self.show_outro()

    def show_intro(self):
        """Show introduction."""
        title = Text(
            "Transformer World Model",
            font_size=48, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "A Deep Dive into the Architecture",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(3)
        self.play(FadeOut(title, subtitle))

    def show_overview(self):
        """Show overview of what will be covered."""
        title = Text("Animation Overview", font_size=32, color=YELLOW)
        title.to_edge(UP, buff=0.8)

        parts = VGroup(
            Text("Part 1: Memory Construction", font_size=20, color=BLUE_C),
            Text("Part 2: Parallel Prediction", font_size=20, color=PURPLE_A),
            Text("Part 3: Prediction Heads", font_size=20, color=GREEN_C),
            Text("Part 4: Training Masking (T-BTF)", font_size=20, color=ORANGE),
        ).arrange(DOWN, buff=0.4)
        parts.move_to(ORIGIN)

        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT) for p in parts], lag_ratio=0.3))

        note = Text(
            "Run individual scenes for full animations",
            font_size=16, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note))

        self.wait(3)
        self.play(FadeOut(title, parts, note))

    def show_outro(self):
        """Show outro."""
        title = Text(
            "Transformer World Model",
            font_size=42, color=GREEN_C, weight="BOLD"
        )

        summary = VGroup(
            Text("Key Components:", font_size=24, color=WHITE),
            Text("• Memory: Encoded history + positional encoding", font_size=18, color=GREY_B),
            Text("• Queries: Learnable tokens for parallel prediction", font_size=18, color=GREY_B),
            Text("• Decoder: Self-attention + Cross-attention layers", font_size=18, color=GREY_B),
            Text("• Heads: Next state, reward, and done prediction", font_size=18, color=GREY_B),
            Text("• T-BTF: Efficient parallel training with masking", font_size=18, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        title.to_edge(UP, buff=1)
        summary.next_to(title, DOWN, buff=0.8)

        self.play(FadeIn(title, shift=DOWN))
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT) for s in summary], lag_ratio=0.2))

        self.wait(5)
        self.play(FadeOut(*self.mobjects))
