"""
Scene 3: Stochastic State (RSSM)
Shows the stochastic latent variable prediction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from manim import (
    Scene, VGroup, Text, Square, RoundedRectangle, Rectangle, Arrow,
    Circle, Polygon, MathTex, Brace,
    WHITE, GREY_B, GREY_D, BLUE_C, YELLOW, GREEN_C, PURPLE_A, ORANGE, RED_C,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart, AnimationGroup,
)
from gru_scenes.common import (
    BACKGROUND_COLOR, GRUColors, styled_text,
    make_embedding_bar, make_mlp_block, make_gaussian_distribution
)


class GRU_WM_StochasticState(Scene):
    """
    Visualizes the stochastic state prediction (RSSM-style):
    - Deterministic state from GRU
    - Stochastic predictor outputs mu and sigma
    - Sampling from Gaussian
    - Combined state
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_title()
        self.show_deterministic_input()
        self.show_stochastic_predictor()
        self.show_sampling()
        self.show_combined_state()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "GRU World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 3: Stochastic State (RSSM)",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_deterministic_input(self):
        """Show the deterministic state from GRU."""
        section_title = styled_text(
            "Step 1: Deterministic State from GRU",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Deterministic state
        self.det_state = Rectangle(
            width=0.5, height=1.8,
            color=GREEN_C,
            fill_opacity=0.5,
            stroke_width=2
        )
        self.det_state.move_to(LEFT * 4 + DOWN * 0.3)
        det_label = styled_text("Deterministic\nState (h_t)\n1024-dim", font_size=12, color=WHITE)
        det_label.next_to(self.det_state, DOWN, buff=0.2)

        # GRU source
        gru_box = RoundedRectangle(
            width=1.5, height=0.8,
            corner_radius=0.1,
            color=PURPLE_A,
            fill_opacity=0.3,
            stroke_width=2
        )
        gru_box.move_to(LEFT * 4 + UP * 2)
        gru_label = styled_text("GRU", font_size=14, color=WHITE)
        gru_label.move_to(gru_box.get_center())

        arrow = Arrow(gru_box.get_bottom(), self.det_state.get_top(),
                      buff=0.15, color=GREEN_C, tip_length=0.12)

        self.play(FadeIn(gru_box, gru_label))
        self.play(GrowArrow(arrow), FadeIn(self.det_state, det_label))

        explanation = styled_text(
            "The deterministic state captures what we know for certain",
            font_size=14, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation))

        self.wait(1.5)
        self.play(FadeOut(gru_box, gru_label, arrow, explanation, section_title))
        self.det_label = det_label

    def show_stochastic_predictor(self):
        """Show the stochastic predictor MLP."""
        section_title = styled_text(
            "Step 2: Stochastic Predictor",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move det state to left
        self.play(
            self.det_state.animate.move_to(LEFT * 5 + DOWN * 0.3),
            self.det_label.animate.move_to(LEFT * 5 + DOWN * 1.5)
        )

        # Stochastic predictor MLP
        predictor = make_mlp_block(
            label="Stochastic\nPredictor",
            width=2.2, height=1.0,
            color=ORANGE
        )
        predictor.move_to(LEFT * 1.5 + DOWN * 0.3)

        arrow1 = Arrow(self.det_state.get_right(), predictor.get_left(),
                       buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow1), FadeIn(predictor))

        # Outputs: mu and log_var
        mu_bar = Rectangle(
            width=0.4, height=1.0,
            color=ORANGE,
            fill_opacity=0.6,
            stroke_width=2
        )
        mu_bar.move_to(RIGHT * 2 + UP * 0.5)
        mu_label = MathTex(r"\mu", font_size=28, color=WHITE)
        mu_label.next_to(mu_bar, UP, buff=0.1)
        mu_dim = styled_text("1024", font_size=10, color=GREY_B)
        mu_dim.next_to(mu_bar, DOWN, buff=0.1)

        logvar_bar = Rectangle(
            width=0.4, height=1.0,
            color=ORANGE,
            fill_opacity=0.4,
            stroke_width=2
        )
        logvar_bar.move_to(RIGHT * 2 + DOWN * 1.2)
        logvar_label = MathTex(r"\log \sigma^2", font_size=22, color=WHITE)
        logvar_label.next_to(logvar_bar, UP, buff=0.1)
        logvar_dim = styled_text("1024", font_size=10, color=GREY_B)
        logvar_dim.next_to(logvar_bar, DOWN, buff=0.1)

        arrow2 = Arrow(predictor.get_right(), mu_bar.get_left() + DOWN * 0.3,
                       buff=0.15, color=ORANGE, tip_length=0.1)
        arrow3 = Arrow(predictor.get_right(), logvar_bar.get_left() + UP * 0.3,
                       buff=0.15, color=ORANGE, tip_length=0.1)

        self.play(
            GrowArrow(arrow2), FadeIn(mu_bar, mu_label, mu_dim),
            GrowArrow(arrow3), FadeIn(logvar_bar, logvar_label, logvar_dim)
        )

        # Explanation
        explanation = styled_text(
            "Predicts mean (μ) and log variance (log σ²) of stochastic state",
            font_size=14, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation))

        self.wait(1.5)

        # Store for next step
        self.mu_bar = mu_bar
        self.mu_label = mu_label
        self.logvar_bar = logvar_bar
        self.logvar_label = logvar_label
        self.predictor = predictor

        self.play(FadeOut(explanation, arrow1, arrow2, arrow3, mu_dim, logvar_dim, section_title))

    def show_sampling(self):
        """Show sampling from Gaussian with reparameterization trick."""
        section_title = styled_text(
            "Step 3: Reparameterization Trick",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Rearrange mu and logvar
        self.play(
            self.det_state.animate.move_to(LEFT * 6 + DOWN * 0.3).scale(0.7),
            self.det_label.animate.move_to(LEFT * 6 + DOWN * 1.3).scale(0.8),
            self.predictor.animate.move_to(LEFT * 3.5 + DOWN * 0.3).scale(0.8),
            self.mu_bar.animate.move_to(LEFT * 0.5 + UP * 0.8).scale(0.8),
            self.mu_label.animate.move_to(LEFT * 0.5 + UP * 1.6).scale(0.8),
            self.logvar_bar.animate.move_to(LEFT * 0.5 + DOWN * 0.8).scale(0.8),
            self.logvar_label.animate.move_to(LEFT * 0.5 + DOWN * 0.1).scale(0.7),
        )

        # Gaussian distribution visualization
        gaussian = make_gaussian_distribution(width=1.5, height=0.8, color=ORANGE)
        gaussian.move_to(RIGHT * 2.5 + UP * 0.3)

        self.play(Create(gaussian))

        # Formula
        formula = MathTex(
            r"z = \mu + \sigma \cdot \epsilon",
            font_size=24, color=WHITE
        )
        formula.move_to(RIGHT * 2.5 + DOWN * 1.2)

        epsilon_note = MathTex(
            r"\epsilon \sim \mathcal{N}(0, I)",
            font_size=18, color=GREY_B
        )
        epsilon_note.next_to(formula, DOWN, buff=0.15)

        self.play(Write(formula), FadeIn(epsilon_note))

        # Stochastic state output (same size as deterministic state after scaling: 0.5*0.7=0.35, 1.8*0.7=1.26)
        stoch_state = Rectangle(
            width=0.35, height=1.26,
            color=ORANGE,
            fill_opacity=0.5,
            stroke_width=2
        )
        stoch_state.move_to(RIGHT * 5.5 + DOWN * 0.3)
        stoch_label = styled_text("Stochastic\nState (z)\n1024-dim", font_size=10, color=WHITE)
        stoch_label.next_to(stoch_state, DOWN, buff=0.15)

        arrow = Arrow(gaussian.get_right(), stoch_state.get_left(),
                      buff=0.15, color=ORANGE, tip_length=0.12)
        sample_label = styled_text("sample", font_size=10, color=ORANGE)
        sample_label.next_to(arrow, UP, buff=0.05)

        self.play(GrowArrow(arrow), FadeIn(sample_label), FadeIn(stoch_state, stoch_label))

        # Explanation
        explanation = VGroup(
            styled_text("Reparameterization allows gradients to flow through sampling",
                        font_size=13, color=GREY_B),
            styled_text("σ = exp(0.5 × log σ²)", font_size=11, color=GREY_B),
        ).arrange(DOWN, buff=0.1)
        explanation.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(explanation))

        self.wait(2)

        # Store for next step
        self.stoch_state = stoch_state
        self.stoch_label = stoch_label

        self.play(FadeOut(
            gaussian, formula, epsilon_note, arrow, sample_label,
            self.predictor, self.mu_bar, self.mu_label,
            self.logvar_bar, self.logvar_label, explanation, section_title
        ))

    def show_combined_state(self):
        """Show concatenation of deterministic and stochastic states."""
        section_title = styled_text(
            "Step 4: Combined State",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Rearrange det and stoch states (scale both to same size)
        self.play(
            self.det_state.animate.scale(1.3).move_to(LEFT * 3 + DOWN * 0.3),
            self.det_label.animate.scale(1.1).move_to(LEFT * 3 + DOWN * 1.6),
            self.stoch_state.animate.scale(1.3).move_to(LEFT * 0.5 + DOWN * 0.3),
            self.stoch_label.animate.scale(1.1).move_to(LEFT * 0.5 + DOWN * 1.6),
        )

        # Plus sign
        plus = styled_text("+", font_size=32, color=WHITE)
        plus.move_to(LEFT * 1.7 + DOWN * 0.3)
        self.play(FadeIn(plus))

        # Concatenation symbol
        concat = styled_text("concat", font_size=14, color=YELLOW)
        concat.move_to(RIGHT * 1.5 + DOWN * 0.3)
        self.play(FadeIn(concat))

        # Combined state
        combined_state = Rectangle(
            width=0.5, height=2.4,
            color=PURPLE_A,
            fill_opacity=0.5,
            stroke_width=2
        )
        combined_state.move_to(RIGHT * 4 + DOWN * 0.3)

        # Color gradient to show both parts
        det_part = Rectangle(
            width=0.45, height=1.15,
            color=GREEN_C,
            fill_opacity=0.4,
            stroke_width=0
        )
        det_part.move_to(combined_state.get_center() + UP * 0.6)

        stoch_part = Rectangle(
            width=0.45, height=1.15,
            color=ORANGE,
            fill_opacity=0.4,
            stroke_width=0
        )
        stoch_part.move_to(combined_state.get_center() + DOWN * 0.6)

        combined_label = styled_text("Combined\nState\n2048-dim", font_size=12, color=WHITE)
        combined_label.next_to(combined_state, DOWN, buff=0.2)

        # Brace
        brace = Brace(combined_state, RIGHT, color=WHITE)
        brace_label = styled_text("det + stoch", font_size=10, color=GREY_B)
        brace_label.next_to(brace, RIGHT, buff=0.1)

        arrow = Arrow(concat.get_right(), combined_state.get_left(),
                      buff=0.15, color=WHITE, tip_length=0.12)

        self.play(
            GrowArrow(arrow),
            FadeIn(combined_state, det_part, stoch_part, combined_label)
        )
        self.play(FadeIn(brace, brace_label))

        # Summary
        summary = VGroup(
            styled_text("Combined state captures both:", font_size=14, color=WHITE),
            styled_text("• Deterministic: What we know for certain", font_size=12, color=GREEN_C),
            styled_text("• Stochastic: Uncertainty and variability", font_size=12, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        summary.to_edge(DOWN, buff=0.6)

        self.play(LaggedStart(*[FadeIn(s, shift=LEFT) for s in summary], lag_ratio=0.2))

        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
