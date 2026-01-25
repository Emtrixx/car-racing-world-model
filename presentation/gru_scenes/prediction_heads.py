"""
Scene 4: Prediction Heads
Shows how predictions are made from the combined state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import (
    Scene, VGroup, Text, Square, RoundedRectangle, Rectangle, Arrow,
    WHITE, GREY_B, GREY_D, BLUE_C, YELLOW, GREEN_C, PURPLE_A, ORANGE, RED_C,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart,
)
from gru_scenes.common import (
    BACKGROUND_COLOR, GRUColors, styled_text,
    make_embedding_bar, make_token_grid, make_mlp_block, make_prediction_head
)


class GRU_WM_PredictionHeads(Scene):
    """
    Visualizes the prediction heads:
    - Decoder head: next token logits
    - Reward head: scalar reward
    - Done head: termination probability
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_title()
        self.show_combined_state_input()
        self.show_decoder_head()
        self.show_auxiliary_heads()
        self.show_summary()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "GRU World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 4: Prediction Heads",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_combined_state_input(self):
        """Show the combined state as input."""
        section_title = styled_text(
            "Step 1: Combined State Input",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Combined state
        self.combined_state = Rectangle(
            width=0.5, height=2.0,
            color=PURPLE_A,
            fill_opacity=0.5,
            stroke_width=2
        )
        self.combined_state.move_to(LEFT * 5 + DOWN * 0.3)

        # Show det + stoch parts
        det_part = Rectangle(
            width=0.45, height=0.95,
            color=GREEN_C, fill_opacity=0.4, stroke_width=0
        )
        det_part.move_to(self.combined_state.get_center() + UP * 0.5)

        stoch_part = Rectangle(
            width=0.45, height=0.95,
            color=ORANGE, fill_opacity=0.4, stroke_width=0
        )
        stoch_part.move_to(self.combined_state.get_center() + DOWN * 0.5)

        combined_label = styled_text("Combined State\n(2048-dim)", font_size=12, color=WHITE)
        combined_label.next_to(self.combined_state, DOWN, buff=0.2)

        self.play(FadeIn(self.combined_state, det_part, stoch_part, combined_label))

        # Explanation
        explanation = styled_text(
            "Combined state (deterministic + stochastic) feeds into all prediction heads",
            font_size=14, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation))

        self.wait(1.5)

        # Store parts
        self.det_part = det_part
        self.stoch_part = stoch_part
        self.combined_label = combined_label

        self.play(FadeOut(explanation, section_title))

    def show_decoder_head(self):
        """Show the decoder head for next token prediction."""
        section_title = styled_text(
            "Step 2: Decoder Head (Next State)",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Decoder head MLP
        decoder = make_mlp_block(
            label="Decoder Head\nLinear+LN+ReLU+Linear",
            width=2.8, height=1.0,
            color=GREEN_C
        )
        decoder.move_to(LEFT * 1 + UP * 1)

        arrow1 = Arrow(
            self.combined_state.get_right() + UP * 0.5,
            decoder.get_left(),
            buff=0.15, color=WHITE, tip_length=0.12
        )

        self.play(GrowArrow(arrow1), FadeIn(decoder))

        # Output logits grid
        logits_label = styled_text("Output: 16 × 512 logits", font_size=12, color=WHITE)
        logits_label.move_to(RIGHT * 3.5 + UP * 2)

        # Create 4x4 grid of logit bars
        logits_grid = VGroup()
        for i in range(16):
            bar = RoundedRectangle(
                width=0.4, height=0.15,
                corner_radius=0.03,
                color=GREEN_C,
                fill_opacity=0.5,
                stroke_width=1
            )
            logits_grid.add(bar)
        logits_grid.arrange_in_grid(rows=4, cols=4, buff=0.05)
        logits_grid.move_to(RIGHT * 3.5 + UP * 0.8)

        per_token_label = styled_text("Each: 512 classes", font_size=10, color=GREY_B)
        per_token_label.next_to(logits_grid, DOWN, buff=0.15)

        arrow2 = Arrow(decoder.get_right(), logits_grid.get_left(),
                      buff=0.15, color=GREEN_C, tip_length=0.12)

        self.play(GrowArrow(arrow2), FadeIn(logits_label, logits_grid, per_token_label))

        # Softmax + argmax to get tokens
        softmax_label = styled_text("softmax → argmax", font_size=11, color=YELLOW)
        softmax_label.next_to(logits_grid, RIGHT, buff=0.3)

        # Predicted tokens
        pred_tokens = make_token_grid(
            rows=4, cols=4, cell_size=0.35,
            color=GREEN_C, show_indices=False
        )
        pred_tokens.move_to(RIGHT * 5.5 + DOWN * 1.5)
        pred_label = styled_text("Predicted\nNext Tokens", font_size=10, color=WHITE)
        pred_label.next_to(pred_tokens, DOWN, buff=0.15)

        arrow3 = Arrow(logits_grid.get_bottom() + DOWN * 0.3, pred_tokens.get_top(),
                      buff=0.1, color=GREEN_C, tip_length=0.1)

        self.play(FadeIn(softmax_label))
        self.play(GrowArrow(arrow3), FadeIn(pred_tokens, pred_label))

        self.wait(1.5)

        # Store for later
        self.decoder = decoder
        self.logits_grid = logits_grid
        self.pred_tokens = pred_tokens

        self.play(FadeOut(
            arrow1, arrow2, arrow3, logits_label, per_token_label,
            softmax_label, pred_label, section_title
        ))

    def show_auxiliary_heads(self):
        """Show reward and done prediction heads."""
        section_title = styled_text(
            "Step 3: Reward & Done Heads",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Move existing elements
        self.play(
            self.decoder.animate.scale(0.7).move_to(LEFT * 1 + UP * 1.8),
            self.logits_grid.animate.scale(0.6).move_to(RIGHT * 2 + UP * 1.8),
            self.pred_tokens.animate.scale(0.6).move_to(RIGHT * 4.5 + UP * 1.8),
        )

        # Reward head
        reward_head = make_prediction_head(
            label="Reward Head\n(Linear)",
            width=2.0, height=0.7,
            color=YELLOW
        )
        reward_head.move_to(LEFT * 1 + DOWN * 0.3)

        arrow_r1 = Arrow(
            self.combined_state.get_right(),
            reward_head.get_left(),
            buff=0.15, color=WHITE, tip_length=0.12
        )

        reward_output = styled_text("+0.42", font_size=20, color=YELLOW, weight="BOLD")
        reward_output.move_to(RIGHT * 2.5 + DOWN * 0.3)
        reward_label = styled_text("Predicted Reward", font_size=11, color=GREY_B)
        reward_label.next_to(reward_output, DOWN, buff=0.1)

        arrow_r2 = Arrow(reward_head.get_right(), reward_output.get_left(),
                        buff=0.2, color=YELLOW, tip_length=0.12)

        self.play(GrowArrow(arrow_r1), FadeIn(reward_head))
        self.play(GrowArrow(arrow_r2), FadeIn(reward_output, reward_label))

        # Done head
        done_head = make_prediction_head(
            label="Done Head\n(Linear)",
            width=2.0, height=0.7,
            color=RED_C
        )
        done_head.move_to(LEFT * 1 + DOWN * 1.8)

        arrow_d1 = Arrow(
            self.combined_state.get_right() + DOWN * 0.5,
            done_head.get_left(),
            buff=0.15, color=WHITE, tip_length=0.12
        )

        done_output = styled_text("0.03", font_size=20, color=RED_C, weight="BOLD")
        done_output.move_to(RIGHT * 2.5 + DOWN * 1.8)
        done_label = styled_text("Termination Prob", font_size=11, color=GREY_B)
        done_label.next_to(done_output, DOWN, buff=0.1)

        arrow_d2 = Arrow(done_head.get_right(), done_output.get_left(),
                        buff=0.2, color=RED_C, tip_length=0.12)

        self.play(GrowArrow(arrow_d1), FadeIn(done_head))
        self.play(GrowArrow(arrow_d2), FadeIn(done_output, done_label))

        # Sigmoid note
        sigmoid_note = styled_text(
            "Done output: sigmoid → [0, 1] probability",
            font_size=12, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(sigmoid_note))

        self.wait(2)
        self.play(FadeOut(*self.mobjects))

    def show_summary(self):
        """Show summary of all predictions."""
        title = styled_text(
            "GRU World Model Predictions",
            font_size=32, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Summary diagram
        combined = Rectangle(
            width=0.6, height=1.5,
            color=PURPLE_A,
            fill_opacity=0.5,
            stroke_width=2
        )
        combined.move_to(LEFT * 4)
        combined_label = styled_text("Combined\nState", font_size=11, color=WHITE)
        combined_label.next_to(combined, DOWN, buff=0.15)

        # Three outputs
        outputs = VGroup(
            VGroup(
                RoundedRectangle(width=1.5, height=0.5, corner_radius=0.1,
                               color=GREEN_C, fill_opacity=0.3, stroke_width=2),
                styled_text("Next State", font_size=12, color=GREEN_C)
            ),
            VGroup(
                RoundedRectangle(width=1.5, height=0.5, corner_radius=0.1,
                               color=YELLOW, fill_opacity=0.3, stroke_width=2),
                styled_text("Reward", font_size=12, color=YELLOW)
            ),
            VGroup(
                RoundedRectangle(width=1.5, height=0.5, corner_radius=0.1,
                               color=RED_C, fill_opacity=0.3, stroke_width=2),
                styled_text("Done", font_size=12, color=RED_C)
            ),
        )

        for i, out in enumerate(outputs):
            out[1].move_to(out[0].get_center())
            out.move_to(RIGHT * 2 + UP * (1 - i))

        outputs.move_to(RIGHT * 1)

        # Arrows
        arrows = VGroup()
        for i, out in enumerate(outputs):
            arr = Arrow(combined.get_right(), out[0].get_left(),
                       buff=0.15, color=WHITE, tip_length=0.1)
            arrows.add(arr)

        self.play(FadeIn(combined, combined_label))
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
            LaggedStart(*[FadeIn(o) for o in outputs], lag_ratio=0.1)
        )

        # Output descriptions
        descriptions = VGroup(
            styled_text("16 × 512 logits → 4×4 token indices", font_size=11, color=GREY_B),
            styled_text("Scalar reward value", font_size=11, color=GREY_B),
            styled_text("Episode termination probability", font_size=11, color=GREY_B),
        )
        for i, desc in enumerate(descriptions):
            desc.next_to(outputs[i], RIGHT, buff=0.3)

        self.play(LaggedStart(*[FadeIn(d, shift=LEFT) for d in descriptions], lag_ratio=0.1))

        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
