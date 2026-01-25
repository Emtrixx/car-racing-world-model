"""
Scene 1: State Encoding
Shows how observations are encoded into a state embedding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import (
    Scene, VGroup, Text, Square, RoundedRectangle, Arrow, MathTex,
    WHITE, GREY_B, BLUE_C, BLUE_D, YELLOW, GREEN_C, PURPLE_A,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart,
)
from gru_scenes.common import (
    BACKGROUND_COLOR, GRUColors, styled_text,
    make_embedding_bar, make_token_grid, make_mlp_block, make_state_vector
)


class GRU_WM_StateEncoding(Scene):
    """
    Visualizes how the GRU World Model encodes observations:
    Token Grid → Token Embedding → Position Embedding → Flatten → Encoder → State
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_title()
        self.show_token_input()
        self.show_token_embedding()
        self.show_position_embedding()
        self.show_encoder()
        self.show_output()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "GRU World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 1: State Encoding",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_token_input(self):
        """Show the 4×4 token grid input."""
        section_title = styled_text(
            "Step 1: Token Grid Input",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create token grid with sample indices
        indices = [127, 43, 512, 89, 234, 67, 401, 156,
                   312, 78, 445, 23, 189, 356, 267, 98]
        self.token_grid = make_token_grid(
            rows=4, cols=4, cell_size=0.5,
            color=GRUColors.token_fill,
            show_indices=True, indices=indices
        )
        self.token_grid.move_to(LEFT * 4 + DOWN * 0.3)

        grid_label = styled_text("4×4 VQ-VAE Tokens", font_size=16, color=WHITE)
        grid_label.next_to(self.token_grid, UP, buff=0.3)

        # Explanation
        explanation = styled_text(
            "Each cell contains an index into a codebook of 512 embeddings",
            font_size=14, color=GREY_B
        ).to_edge(DOWN, buff=0.8)

        self.play(FadeIn(self.token_grid, grid_label, shift=LEFT))
        self.play(FadeIn(explanation))
        self.wait(1.5)

        # Store for later
        self.grid_label = grid_label
        self.play(FadeOut(explanation, section_title))

    def show_token_embedding(self):
        """Show token embedding lookup."""
        section_title = styled_text(
            "Step 2: Token Embedding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move grid to left
        self.play(
            self.token_grid.animate.scale(0.7).move_to(LEFT * 5.5 + DOWN * 0.3),
            self.grid_label.animate.scale(0.8).move_to(LEFT * 5.5 + UP * 1)
        )

        # Create embedding lookup block
        codebook = RoundedRectangle(
            width=1.5, height=2.0,
            corner_radius=0.1,
            color=BLUE_D,
            fill_opacity=0.3,
            stroke_width=2
        )
        codebook.move_to(LEFT * 2.5 + DOWN * 0.3)
        codebook_label = styled_text("Codebook\n512 × 256", font_size=12, color=WHITE)
        codebook_label.move_to(codebook.get_center())
        codebook_group = VGroup(codebook, codebook_label)

        arrow1 = Arrow(self.token_grid.get_right(), codebook.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow1), FadeIn(codebook_group))

        # Create embedded tokens visualization
        self.embedded_tokens = VGroup()
        for i in range(16):
            bar = make_embedding_bar(width=0.5, height=0.1, gradient=True)
            self.embedded_tokens.add(bar)
        self.embedded_tokens.arrange_in_grid(rows=4, cols=4, buff=0.04)
        self.embedded_tokens.move_to(RIGHT * 0.5 + DOWN * 0.3)

        embed_label = styled_text("Token Embeddings\n16 × 256-dim", font_size=12, color=WHITE)
        embed_label.next_to(self.embedded_tokens, UP, buff=0.2)

        arrow2 = Arrow(codebook.get_right(), self.embedded_tokens.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow2), FadeIn(self.embedded_tokens, embed_label))

        # Show lookup animation
        self.play(
            Indicate(self.token_grid[0], color=YELLOW),
            Indicate(self.embedded_tokens[0], color=YELLOW),
            run_time=0.8
        )

        self.wait(1)
        self.play(FadeOut(arrow1, arrow2, codebook_group, section_title, self.grid_label))
        self.embed_label = embed_label

    def show_position_embedding(self):
        """Show grid positional embedding being added."""
        section_title = styled_text(
            "Step 3: Grid Position Embedding",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move token embeddings to left
        self.play(
            self.token_grid.animate.move_to(LEFT * 6 + DOWN * 0.3),
            self.embedded_tokens.animate.move_to(LEFT * 3.5 + DOWN * 0.3),
            self.embed_label.animate.move_to(LEFT * 3.5 + UP * 0.8)
        )

        # Create position embeddings (single color - unique per position)
        pos_embeds = VGroup()
        for i in range(16):
            bar = RoundedRectangle(
                width=0.5, height=0.1,
                corner_radius=0.03,
                color=YELLOW,
                fill_opacity=0.5,
                stroke_width=1.5
            )
            pos_embeds.add(bar)
        pos_embeds.arrange_in_grid(rows=4, cols=4, buff=0.04)
        pos_embeds.move_to(LEFT * 0.3 + DOWN * 0.3)

        pos_label = styled_text("Position\nEmbeddings", font_size=12, color=WHITE)
        pos_label.next_to(pos_embeds, UP, buff=0.2)

        # Plus sign
        plus = styled_text("+", font_size=28, color=WHITE)
        plus.move_to(LEFT * 1.8 + DOWN * 0.3)

        self.play(FadeIn(plus), FadeIn(pos_embeds, pos_label))

        # Equals sign
        equals = styled_text("=", font_size=28, color=WHITE)
        equals.move_to(RIGHT * 1.5 + DOWN * 0.3)
        self.play(FadeIn(equals))

        # Result embeddings with position info
        self.result_embeds = VGroup()
        for i in range(16):
            bar = make_embedding_bar(width=0.5, height=0.1, gradient=True)
            bar[1].set_stroke(YELLOW, width=2)
            self.result_embeds.add(bar)
        self.result_embeds.arrange_in_grid(rows=4, cols=4, buff=0.04)
        self.result_embeds.move_to(RIGHT * 3.5 + DOWN * 0.3)

        result_label = styled_text("Positioned\nEmbeddings", font_size=12, color=WHITE)
        result_label.next_to(self.result_embeds, UP, buff=0.2)

        self.play(FadeIn(self.result_embeds, result_label))

        self.wait(1.5)
        self.play(FadeOut(
            self.token_grid, self.embedded_tokens, self.embed_label,
            pos_embeds, pos_label, plus, equals, section_title
        ))
        self.result_label = result_label

    def show_encoder(self):
        """Show the encoder MLP that processes flattened embeddings."""
        section_title = styled_text(
            "Step 4: Flatten & Encode",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move result embeddings to left
        self.play(
            self.result_embeds.animate.move_to(LEFT * 4.5 + DOWN * 0.3),
            self.result_label.animate.move_to(LEFT * 4.5 + UP * 0.8)
        )

        # Flatten visualization
        flat_bar = RoundedRectangle(
            width=3.0, height=0.25,
            corner_radius=0.1,
            color=BLUE_C,
            fill_opacity=0.5,
            stroke_width=2
        )
        flat_bar.move_to(LEFT * 1 + DOWN * 0.3)
        flat_label = styled_text("Flattened: 16×256 = 4096-dim", font_size=12, color=WHITE)
        flat_label.next_to(flat_bar, UP, buff=0.15)

        arrow1 = Arrow(self.result_embeds.get_right(), flat_bar.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow1), FadeIn(flat_bar, flat_label))

        # Encoder MLP
        encoder = make_mlp_block(
            label="Encoder\nLinear+ReLU+LN",
            width=2.0, height=1.0,
            color=GREEN_C
        )
        encoder.move_to(RIGHT * 2.5 + DOWN * 0.3)

        arrow2 = Arrow(flat_bar.get_right(), encoder.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow2), FadeIn(encoder))

        # Output state embedding
        self.state_embed = make_embedding_bar(width=1.5, height=0.3, gradient=True)
        self.state_embed[1].set_stroke(GREEN_C, width=2)
        self.state_embed.move_to(RIGHT * 5.5 + DOWN * 0.3)
        state_label = styled_text("State Embedding\n1024-dim", font_size=12, color=WHITE)
        state_label.next_to(self.state_embed, UP, buff=0.15)

        arrow3 = Arrow(encoder.get_right(), self.state_embed.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow3), FadeIn(self.state_embed, state_label))

        self.wait(1.5)
        self.play(FadeOut(
            self.result_embeds, self.result_label, flat_bar, flat_label,
            arrow1, arrow2, arrow3, encoder, section_title
        ))
        self.state_label = state_label

    def show_output(self):
        """Show the final output state embedding."""
        title = styled_text(
            "Observation Encoded",
            font_size=32, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Center the state embedding
        self.play(
            self.state_embed.animate.move_to(ORIGIN).scale(1.5),
            self.state_label.animate.move_to(DOWN * 1.2)
        )

        # Summary
        summary = VGroup(
            styled_text("4×4 tokens → 16×256 embeddings → 4096-dim → 1024-dim state",
                       font_size=16, color=GREY_B),
            styled_text("Ready for recurrent processing", font_size=14, color=YELLOW),
        ).arrange(DOWN, buff=0.2)
        summary.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(summary))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
