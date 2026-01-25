"""
Scene 5: GRU World Model Overview
Complete video combining all parts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import (
    Scene, VGroup, Text, Square, RoundedRectangle, Rectangle, Arrow, CurvedArrow,
    Circle, MathTex, Brace, Polygon,
    WHITE, GREY_B, GREY_D, BLUE_C, BLUE_D, TEAL_C, YELLOW, GREEN_C, PURPLE_A, ORANGE, RED_C,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart, AnimationGroup,
)
from gru_scenes.common import (
    BACKGROUND_COLOR, GRUColors, styled_text,
    make_embedding_bar, make_token_grid, make_stacked_gru, make_mlp_block,
    make_gaussian_distribution, make_prediction_head
)


class GRU_WM_Overview(Scene):
    """
    Complete scene combining all parts of the GRU World Model visualization.
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # Introduction
        self.show_intro()

        # Part 1: State Encoding
        self.part1_state_encoding()

        # Part 2: Recurrent Core
        self.part2_recurrent_core()

        # Part 3: Stochastic State
        self.part3_stochastic_state()

        # Part 4: Prediction Heads
        self.part4_prediction_heads()

        # Outro
        self.show_outro()

    def show_intro(self):
        """Show introduction."""
        title = Text(
            "GRU World Model",
            font_size=48, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Recurrent State Space Model",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))

    def show_part_title(self, part_num: int, title_text: str, color):
        """Show a part title card."""
        part_label = Text(f"Part {part_num}", font_size=24, color=GREY_B)
        title = Text(title_text, font_size=36, color=color, weight="BOLD")
        title.next_to(part_label, DOWN, buff=0.3)
        group = VGroup(part_label, title)
        group.move_to(ORIGIN)

        self.play(FadeIn(group, shift=UP))
        self.wait(1.5)
        self.play(FadeOut(group))

    # =========================================================================
    # Part 1: State Encoding
    # =========================================================================
    def part1_state_encoding(self):
        """Visualize state encoding pipeline."""
        self.show_part_title(1, "State Encoding", BLUE_C)

        section_title = styled_text("Observation → State Embedding", font_size=28, color=YELLOW)
        section_title.to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Token grid
        indices = [127, 43, 512, 89, 234, 67, 401, 156,
                   312, 78, 445, 23, 189, 356, 267, 98]
        token_grid = make_token_grid(
            rows=4, cols=4, cell_size=0.4,
            color=GRUColors.token_fill,
            show_indices=True, indices=indices
        )
        token_grid.move_to(LEFT * 5 + DOWN * 0.3)
        grid_label = styled_text("4×4 Tokens", font_size=12, color=WHITE)
        grid_label.next_to(token_grid, UP, buff=0.2)

        self.play(FadeIn(token_grid, grid_label))

        # Embedding lookup
        codebook = RoundedRectangle(
            width=1.2, height=1.5, corner_radius=0.1,
            color=BLUE_D, fill_opacity=0.3, stroke_width=2
        )
        codebook.move_to(LEFT * 2.5 + DOWN * 0.3)
        cb_label = styled_text("Codebook\n512×256", font_size=10, color=WHITE)
        cb_label.move_to(codebook.get_center())

        arrow1 = Arrow(token_grid.get_right(), codebook.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.1)

        self.play(GrowArrow(arrow1), FadeIn(codebook, cb_label))

        # Embedded tokens + position
        embeds = VGroup()
        for i in range(16):
            bar = make_embedding_bar(width=0.35, height=0.08, gradient=True)
            bar[1].set_stroke(YELLOW, width=1.5)
            embeds.add(bar)
        embeds.arrange_in_grid(rows=4, cols=4, buff=0.03)
        embeds.move_to(RIGHT * 0.3 + DOWN * 0.3)
        embed_label = styled_text("+ Position\nEmbeddings", font_size=10, color=WHITE)
        embed_label.next_to(embeds, UP, buff=0.15)

        arrow2 = Arrow(codebook.get_right(), embeds.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.1)

        self.play(GrowArrow(arrow2), FadeIn(embeds, embed_label))

        # Encoder MLP
        encoder = make_mlp_block(label="Encoder\nMLP", width=1.5, height=0.8, color=GREEN_C)
        encoder.move_to(RIGHT * 3 + DOWN * 0.3)

        arrow3 = Arrow(embeds.get_right(), encoder.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.1)

        self.play(GrowArrow(arrow3), FadeIn(encoder))

        # State embedding output
        state_embed = make_embedding_bar(width=1.0, height=0.25, gradient=True)
        state_embed[1].set_stroke(GREEN_C, width=2)
        state_embed.move_to(RIGHT * 5.5 + DOWN * 0.3)
        state_label = styled_text("State\n1024-dim", font_size=10, color=WHITE)
        state_label.next_to(state_embed, DOWN, buff=0.1)

        arrow4 = Arrow(encoder.get_right(), state_embed.get_left(),
                       buff=0.15, color=GREEN_C, tip_length=0.1)

        self.play(GrowArrow(arrow4), FadeIn(state_embed, state_label))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    # =========================================================================
    # Part 2: Recurrent Core
    # =========================================================================
    def part2_recurrent_core(self):
        """Visualize GRU recurrent processing."""
        self.show_part_title(2, "Recurrent Core", PURPLE_A)

        section_title = styled_text("GRU: Sequential State Updates", font_size=28, color=YELLOW)
        section_title.to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create 3 timesteps
        timesteps = VGroup()
        hidden_arrows = VGroup()

        for t in range(3):
            # GRU cell
            gru_cell = RoundedRectangle(
                width=1.5, height=1.0,
                corner_radius=0.1, color=PURPLE_A,
                fill_opacity=0.3, stroke_width=2
            )
            gru_label = styled_text("GRU", font_size=14, color=WHITE, weight="BOLD")
            gru_label.move_to(gru_cell.get_center())

            # Inputs
            obs_input = Square(side_length=0.2, color=GREEN_C, fill_opacity=0.5)
            act_input = Square(side_length=0.2, color=TEAL_C, fill_opacity=0.5)
            inputs = VGroup(obs_input, act_input).arrange(RIGHT, buff=0.05)
            inputs.next_to(gru_cell, DOWN, buff=0.25)

            # Output
            output = Square(side_length=0.25, color=GREEN_C, fill_opacity=0.6)
            output.next_to(gru_cell, UP, buff=0.25)
            out_label = styled_text(f"h_{t}", font_size=11, color=GREY_B)
            out_label.next_to(output, UP, buff=0.05)

            # Time label
            time_label = styled_text(f"t={t}", font_size=12, color=YELLOW)
            time_label.next_to(inputs, DOWN, buff=0.15)

            timestep = VGroup(gru_cell, gru_label, inputs, output, out_label, time_label)
            timestep.move_to(LEFT * 4 + RIGHT * t * 3)
            timesteps.add(timestep)

            # Hidden state arrow
            if t < 2:
                h_arrow = Arrow(
                    gru_cell.get_right() + UP * 0.2,
                    gru_cell.get_right() + RIGHT * 1.5 + UP * 0.2,
                    color=PURPLE_A, stroke_width=2, tip_length=0.1
                )
                h_arrow.shift(LEFT * 4 + RIGHT * t * 3)
                hidden_arrows.add(h_arrow)

        self.play(LaggedStart(*[FadeIn(ts, shift=UP) for ts in timesteps], lag_ratio=0.2))
        self.play(LaggedStart(*[GrowArrow(arr) for arr in hidden_arrows], lag_ratio=0.15))

        # Legend
        legend = VGroup(
            VGroup(Square(side_length=0.15, color=GREEN_C, fill_opacity=0.5),
                   styled_text("obs_embed", font_size=10)).arrange(RIGHT, buff=0.1),
            VGroup(Square(side_length=0.15, color=TEAL_C, fill_opacity=0.5),
                   styled_text("action_embed", font_size=10)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.4).to_edge(DOWN, buff=0.8)

        self.play(FadeIn(legend))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    # =========================================================================
    # Part 3: Stochastic State
    # =========================================================================
    def part3_stochastic_state(self):
        """Visualize stochastic state prediction."""
        self.show_part_title(3, "Stochastic State (RSSM)", ORANGE)

        section_title = styled_text("Modeling Uncertainty", font_size=28, color=YELLOW)
        section_title.to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Deterministic state
        det_state = Rectangle(
            width=0.4, height=1.2, color=GREEN_C,
            fill_opacity=0.5, stroke_width=2
        )
        det_state.move_to(LEFT * 5 + DOWN * 0.3)
        det_label = styled_text("Deterministic\n(h_t)", font_size=10, color=WHITE)
        det_label.next_to(det_state, DOWN, buff=0.15)

        self.play(FadeIn(det_state, det_label))

        # Stochastic predictor
        predictor = make_mlp_block(label="Stochastic\nPredictor", width=2.0, height=0.8, color=ORANGE)
        predictor.move_to(LEFT * 2 + DOWN * 0.3)

        arrow1 = Arrow(det_state.get_right(), predictor.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.1)

        self.play(GrowArrow(arrow1), FadeIn(predictor))

        # Gaussian
        gaussian = make_gaussian_distribution(width=1.2, height=0.6, color=ORANGE)
        gaussian.move_to(RIGHT * 1.5 + DOWN * 0.1)

        arrow2 = Arrow(predictor.get_right(), gaussian.get_left(),
                       buff=0.15, color=ORANGE, tip_length=0.1)

        formula = MathTex(r"z \sim \mathcal{N}(\mu, \sigma^2)", font_size=18, color=WHITE)
        formula.next_to(gaussian, DOWN, buff=0.3)

        self.play(GrowArrow(arrow2), FadeIn(gaussian, formula))

        # Stochastic state
        stoch_state = Rectangle(
            width=0.4, height=1.2, color=ORANGE,
            fill_opacity=0.5, stroke_width=2
        )
        stoch_state.move_to(RIGHT * 4 + DOWN * 0.3)
        stoch_label = styled_text("Stochastic\n(z)", font_size=10, color=WHITE)
        stoch_label.next_to(stoch_state, DOWN, buff=0.15)

        arrow3 = Arrow(gaussian.get_right(), stoch_state.get_left(),
                       buff=0.15, color=ORANGE, tip_length=0.1)
        sample_label = styled_text("sample", font_size=9, color=ORANGE)
        sample_label.next_to(arrow3, UP, buff=0.05)

        self.play(GrowArrow(arrow3), FadeIn(sample_label), FadeIn(stoch_state, stoch_label))

        # Combined state note
        note = styled_text(
            "Combined state = concat(deterministic, stochastic) → 2048-dim",
            font_size=13, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    # =========================================================================
    # Part 4: Prediction Heads
    # =========================================================================
    def part4_prediction_heads(self):
        """Visualize prediction heads."""
        self.show_part_title(4, "Prediction Heads", GREEN_C)

        section_title = styled_text("Predicting Next State, Reward, Done", font_size=28, color=YELLOW)
        section_title.to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Combined state
        combined = Rectangle(
            width=0.5, height=1.8, color=PURPLE_A,
            fill_opacity=0.5, stroke_width=2
        )
        combined.move_to(LEFT * 5 + DOWN * 0.3)

        # Show det + stoch parts
        det_part = Rectangle(width=0.45, height=0.85, color=GREEN_C,
                            fill_opacity=0.4, stroke_width=0)
        det_part.move_to(combined.get_center() + UP * 0.45)
        stoch_part = Rectangle(width=0.45, height=0.85, color=ORANGE,
                              fill_opacity=0.4, stroke_width=0)
        stoch_part.move_to(combined.get_center() + DOWN * 0.45)

        combined_label = styled_text("Combined\n2048-dim", font_size=10, color=WHITE)
        combined_label.next_to(combined, DOWN, buff=0.15)

        self.play(FadeIn(combined, det_part, stoch_part, combined_label))

        # Three prediction heads
        # Decoder head
        decoder = make_prediction_head(label="Decoder\nHead", width=1.6, height=0.6, color=GREEN_C)
        decoder.move_to(LEFT * 1 + UP * 1.2)

        # Token grid output
        tokens = make_token_grid(rows=4, cols=4, cell_size=0.25, color=GREEN_C)
        tokens.move_to(RIGHT * 2.5 + UP * 1.2)
        token_label = styled_text("Next State\n(16 tokens)", font_size=9, color=WHITE)
        token_label.next_to(tokens, RIGHT, buff=0.15)

        arrow1 = Arrow(combined.get_right() + UP * 0.6, decoder.get_left(),
                       buff=0.1, color=WHITE, tip_length=0.1)
        arrow1b = Arrow(decoder.get_right(), tokens.get_left(),
                        buff=0.1, color=GREEN_C, tip_length=0.1)

        self.play(GrowArrow(arrow1), FadeIn(decoder))
        self.play(GrowArrow(arrow1b), FadeIn(tokens, token_label))

        # Reward head
        reward = make_prediction_head(label="Reward\nHead", width=1.6, height=0.6, color=YELLOW)
        reward.move_to(LEFT * 1 + DOWN * 0.3)

        reward_val = styled_text("+0.42", font_size=16, color=YELLOW, weight="BOLD")
        reward_val.move_to(RIGHT * 2.5 + DOWN * 0.3)

        arrow2 = Arrow(combined.get_right(), reward.get_left(),
                       buff=0.1, color=WHITE, tip_length=0.1)
        arrow2b = Arrow(reward.get_right(), reward_val.get_left(),
                        buff=0.15, color=YELLOW, tip_length=0.1)

        self.play(GrowArrow(arrow2), FadeIn(reward))
        self.play(GrowArrow(arrow2b), FadeIn(reward_val))

        # Done head
        done = make_prediction_head(label="Done\nHead", width=1.6, height=0.6, color=RED_C)
        done.move_to(LEFT * 1 + DOWN * 1.8)

        done_val = styled_text("0.03", font_size=16, color=RED_C, weight="BOLD")
        done_val.move_to(RIGHT * 2.5 + DOWN * 1.8)

        arrow3 = Arrow(combined.get_right() + DOWN * 0.6, done.get_left(),
                       buff=0.1, color=WHITE, tip_length=0.1)
        arrow3b = Arrow(done.get_right(), done_val.get_left(),
                        buff=0.15, color=RED_C, tip_length=0.1)

        self.play(GrowArrow(arrow3), FadeIn(done))
        self.play(GrowArrow(arrow3b), FadeIn(done_val))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    def show_outro(self):
        """Show outro."""
        title = Text(
            "GRU World Model",
            font_size=42, color=GREEN_C, weight="BOLD"
        )

        summary = VGroup(
            Text("Key Components:", font_size=24, color=WHITE),
            Text("• State Encoding: Observation → 1024-dim embedding", font_size=18, color=GREY_B),
            Text("• Recurrent Core: 3-layer GRU maintains temporal state", font_size=18, color=GREY_B),
            Text("• Stochastic State: RSSM-style uncertainty modeling", font_size=18, color=GREY_B),
            Text("• Prediction Heads: Next state, reward, done", font_size=18, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        title.to_edge(UP, buff=1)
        summary.next_to(title, DOWN, buff=0.8)

        self.play(FadeIn(title, shift=DOWN))
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT) for s in summary], lag_ratio=0.2))

        # Comparison note
        comparison = Text(
            "Sequential processing (vs. Transformer's parallel attention)",
            font_size=16, color=ORANGE
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(comparison))

        self.wait(3)
        self.play(FadeOut(*self.mobjects))
