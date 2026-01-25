"""
Scene 2: Recurrent Core
Shows the GRU processing observations and actions over time.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from manim import (
    Scene, VGroup, Text, Square, RoundedRectangle, Rectangle, Arrow, CurvedArrow,
    WHITE, GREY_B, GREY_D, BLUE_C, TEAL_C, YELLOW, GREEN_C, PURPLE_A, ORANGE,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart, AnimationGroup,
)
from gru_scenes.common import (
    BACKGROUND_COLOR, GRUColors, styled_text,
    make_embedding_bar, make_stacked_gru, make_mlp_block, make_state_vector
)


class GRU_WM_RecurrentCore(Scene):
    """
    Visualizes the GRU recurrent core:
    - Input: obs_embedding + action_embedding
    - GRU layers processing
    - Hidden state flow through time
    """

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_title()
        self.show_input_formation()
        self.show_gru_architecture()
        self.show_temporal_processing()
        self.show_hidden_state_flow()

    def show_title(self):
        """Display scene title."""
        title = Text(
            "GRU World Model",
            font_size=42, color=WHITE, weight="BOLD"
        )
        subtitle = Text(
            "Part 2: Recurrent Core",
            font_size=28, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(FadeIn(title, shift=UP))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(2)
        self.play(FadeOut(title, subtitle))
        self.wait(0.5)

    def show_input_formation(self):
        """Show how obs and action embeddings are concatenated."""
        section_title = styled_text(
            "Step 1: Input Formation",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Observation embedding
        obs_embed = make_embedding_bar(width=1.2, height=0.25, gradient=True)
        obs_embed[1].set_stroke(GREEN_C, width=2)
        obs_embed.move_to(LEFT * 4 + UP * 0.8)
        obs_label = styled_text("obs_embed\n(1024-dim)", font_size=12, color=WHITE)
        obs_label.next_to(obs_embed, LEFT, buff=0.2)

        # Action embedding
        action_embed = make_embedding_bar(width=1.2, height=0.25, gradient=False)
        action_embed[0].set_fill(TEAL_C, opacity=0.5)
        action_embed[1].set_stroke(TEAL_C, width=2)
        action_embed.move_to(LEFT * 4 + DOWN * 0.8)
        action_label = styled_text("action_embed\n(1024-dim)", font_size=12, color=WHITE)
        action_label.next_to(action_embed, LEFT, buff=0.2)

        # Action MLP
        action_mlp = make_mlp_block(
            label="Linear",
            width=1.2, height=0.5,
            color=TEAL_C
        )
        action_mlp.move_to(LEFT * 4 + DOWN * 2.2)
        action_input_label = styled_text("action\n(3-dim)", font_size=10, color=GREY_B)
        action_input_label.next_to(action_mlp, DOWN, buff=0.1)

        self.play(FadeIn(obs_embed, obs_label))
        self.play(FadeIn(action_mlp, action_input_label))

        arrow_action = Arrow(action_mlp.get_top(), action_embed.get_bottom(),
                            buff=0.1, color=TEAL_C, tip_length=0.1)
        self.play(GrowArrow(arrow_action), FadeIn(action_embed, action_label))

        # Concatenation
        concat_symbol = styled_text("concat", font_size=14, color=YELLOW)
        concat_symbol.move_to(LEFT * 1.5 + ORIGIN)

        # Combined input
        combined = Rectangle(
            width=0.4, height=2.0,
            color=PURPLE_A,
            fill_opacity=0.4,
            stroke_width=2
        )
        combined.move_to(RIGHT * 1 + ORIGIN)
        combined_label = styled_text("GRU Input\n(2048-dim)", font_size=12, color=WHITE)
        combined_label.next_to(combined, RIGHT, buff=0.2)

        arrow1 = Arrow(obs_embed.get_right(), concat_symbol.get_left() + UP * 0.3,
                      buff=0.1, color=WHITE, tip_length=0.1)
        arrow2 = Arrow(action_embed.get_right(), concat_symbol.get_left() + DOWN * 0.3,
                      buff=0.1, color=WHITE, tip_length=0.1)
        arrow3 = Arrow(concat_symbol.get_right(), combined.get_left(),
                      buff=0.1, color=WHITE, tip_length=0.1)

        self.play(
            GrowArrow(arrow1), GrowArrow(arrow2),
            FadeIn(concat_symbol)
        )
        self.play(GrowArrow(arrow3), FadeIn(combined, combined_label))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    def show_gru_architecture(self):
        """Show the stacked GRU architecture."""
        section_title = styled_text(
            "Step 2: GRU Architecture",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Stacked GRU
        gru_stack = make_stacked_gru(num_layers=3, width=2.5, height_per_layer=0.8)
        gru_stack.move_to(ORIGIN)

        # Input arrow
        input_bar = Rectangle(
            width=0.3, height=1.0,
            color=PURPLE_A,
            fill_opacity=0.4,
            stroke_width=2
        )
        input_bar.move_to(LEFT * 3.5 + DOWN * 0.5)
        input_label = styled_text("Input\n2048", font_size=10, color=WHITE)
        input_label.next_to(input_bar, DOWN, buff=0.1)

        arrow_in = Arrow(input_bar.get_right(), gru_stack.get_left(),
                        buff=0.15, color=WHITE, tip_length=0.12)

        self.play(FadeIn(input_bar, input_label))
        self.play(GrowArrow(arrow_in), FadeIn(gru_stack, shift=RIGHT))

        # Output arrow
        output_bar = Rectangle(
            width=0.3, height=1.0,
            color=GREEN_C,
            fill_opacity=0.4,
            stroke_width=2
        )
        output_bar.move_to(RIGHT * 3.5 + DOWN * 0.5)
        output_label = styled_text("Output\n1024", font_size=10, color=WHITE)
        output_label.next_to(output_bar, DOWN, buff=0.1)

        arrow_out = Arrow(gru_stack.get_right(), output_bar.get_left(),
                         buff=0.15, color=WHITE, tip_length=0.12)

        self.play(GrowArrow(arrow_out), FadeIn(output_bar, output_label))

        # Hidden state
        hidden_label = styled_text("Hidden State (h)\n3 × 1024-dim", font_size=12, color=PURPLE_A)
        hidden_label.next_to(gru_stack, UP, buff=0.5)

        hidden_arrow = CurvedArrow(
            gru_stack.get_top() + LEFT * 0.5,
            gru_stack.get_top() + RIGHT * 0.5,
            angle=-1.5, color=PURPLE_A,
            stroke_width=2, tip_length=0.12
        )

        self.play(FadeIn(hidden_label), Create(hidden_arrow))

        explanation = styled_text(
            "Hidden state maintains information across timesteps",
            font_size=14, color=GREY_B
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(explanation))

        self.wait(1.5)
        self.play(FadeOut(*self.mobjects))

    def show_temporal_processing(self):
        """Show processing across multiple timesteps."""
        section_title = styled_text(
            "Step 3: Sequential Processing",
            font_size=28, color=YELLOW
        ).to_edge(UP, buff=0.5)
        self.play(Write(section_title))

        # Create 3 timesteps - first build and position them
        timesteps = VGroup()
        gru_cells = []  # Store references for arrow creation

        for t in range(3):
            # GRU cell
            gru_cell = RoundedRectangle(
                width=1.8, height=1.2,
                corner_radius=0.1,
                color=PURPLE_A,
                fill_opacity=0.3,
                stroke_width=2
            )
            gru_label = styled_text("GRU", font_size=16, color=WHITE, weight="BOLD")
            gru_label.move_to(gru_cell.get_center())

            # Input
            input_group = VGroup()
            obs_dot = Square(side_length=0.2, color=GREEN_C, fill_opacity=0.5)
            act_dot = Square(side_length=0.2, color=TEAL_C, fill_opacity=0.5)
            input_group.add(obs_dot, act_dot)
            input_group.arrange(DOWN, buff=0.05)
            input_group.next_to(gru_cell, DOWN, buff=0.3)
            input_label = styled_text(f"x_{t}", font_size=12, color=GREY_B)
            input_label.next_to(input_group, DOWN, buff=0.1)

            # Output
            output_dot = Square(side_length=0.25, color=GREEN_C, fill_opacity=0.6)
            output_dot.next_to(gru_cell, UP, buff=0.3)
            output_label = styled_text(f"h_{t}", font_size=12, color=GREY_B)
            output_label.next_to(output_dot, UP, buff=0.1)

            # Time label
            time_label = styled_text(f"t = {t}", font_size=14, color=YELLOW)
            time_label.next_to(input_label, DOWN, buff=0.3)

            timestep = VGroup(gru_cell, gru_label, input_group, input_label,
                            output_dot, output_label, time_label)
            timestep.move_to(LEFT * 4 + RIGHT * t * 3.5)
            timesteps.add(timestep)
            gru_cells.append(gru_cell)

        # Create arrows AFTER positioning - use the actual positioned gru_cells
        hidden_arrows = VGroup()
        for t in range(2):
            h_arrow = Arrow(
                gru_cells[t].get_right(),
                gru_cells[t + 1].get_left(),
                color=PURPLE_A, stroke_width=2, tip_length=0.12
            )
            hidden_arrows.add(h_arrow)

        # Animate timesteps appearing
        self.play(LaggedStart(*[FadeIn(ts, shift=UP) for ts in timesteps], lag_ratio=0.3))
        self.play(LaggedStart(*[GrowArrow(arr) for arr in hidden_arrows], lag_ratio=0.2))

        # Hidden state label
        h_label = styled_text("Hidden state flows →", font_size=14, color=PURPLE_A)
        h_label.move_to(UP * 2.5)
        self.play(FadeIn(h_label))

        self.wait(2)
        self.play(FadeOut(*self.mobjects))

    def show_hidden_state_flow(self):
        """Highlight the hidden state persistence."""
        title = styled_text(
            "Hidden State: Memory Across Time",
            font_size=32, color=GREEN_C, weight="BOLD"
        ).to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Build hidden state rectangles first, then position, then create arrows
        h_rects = []
        h_groups = VGroup()

        for t in range(4):
            # Hidden state rectangle
            h_rect = Rectangle(
                width=0.4, height=1.0,
                color=PURPLE_A,
                fill_opacity=0.4 + 0.1 * t,
                stroke_width=2
            )
            h_label = styled_text(f"h_{t}", font_size=14, color=WHITE)
            h_label.next_to(h_rect, DOWN, buff=0.1)

            h_group = VGroup(h_rect, h_label)
            h_group.move_to(LEFT * 4.5 + RIGHT * t * 2.5)
            h_groups.add(h_group)
            h_rects.append(h_rect)

        # Create arrows using actual positioned rectangles
        arrows = VGroup()
        for t in range(3):
            arrow = Arrow(
                h_rects[t].get_right(),
                h_rects[t + 1].get_left(),
                color=PURPLE_A, stroke_width=2, tip_length=0.12
            )
            arrows.add(arrow)

        diagram = VGroup(h_groups, arrows)
        diagram.move_to(ORIGIN)
        self.play(FadeIn(diagram))

        # Summary points
        summary = VGroup(
            styled_text("• h_0: Initial state (zeros)", font_size=14, color=GREY_B),
            styled_text("• h_t: Accumulates information from all past observations", font_size=14, color=GREY_B),
            styled_text("• Enables long-term dependencies in predictions", font_size=14, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        summary.to_edge(DOWN, buff=0.8)

        self.play(LaggedStart(*[FadeIn(s, shift=LEFT) for s in summary], lag_ratio=0.2))

        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))
