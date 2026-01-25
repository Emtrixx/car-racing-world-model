"""
Common utilities and components for GRU World Model animations.
"""
import numpy as np
from manim import (
    VGroup, Text, Square, RoundedRectangle, Rectangle, Arrow, CurvedArrow,
    Circle, Line, Polygon, MathTex,
    WHITE, GREY_B, GREY_D, BLUE_C, BLUE_D, BLUE_E, TEAL_C, TEAL_D,
    PURPLE_A, PURPLE_B, ORANGE, YELLOW, GREEN_C, RED_C, GOLD,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    FadeIn, FadeOut, Write, Create, GrowArrow, Transform, Indicate,
    LaggedStart, AnimationGroup,
    Scene, LinearTransformationScene,
)

# Background color matching the Transformer scenes
BACKGROUND_COLOR = "#0f1419"

# Default font for consistent text rendering
DEFAULT_FONT = "Sans"


class GRUColors:
    """Color scheme for GRU World Model visualizations."""
    token_fill = BLUE_C
    token_stroke = BLUE_D
    action_fill = TEAL_C
    action_stroke = TEAL_D
    hidden_state = PURPLE_A
    deterministic = GREEN_C
    stochastic = ORANGE
    embedding_gradient = (BLUE_E, BLUE_C)
    text = WHITE

    # Prediction colors
    pred_next_state = GREEN_C
    pred_reward = YELLOW
    pred_done = RED_C


def styled_text(text: str, font_size: int = 16, color=WHITE, weight: str = "NORMAL", **kwargs) -> Text:
    """Create text with consistent styling."""
    return Text(text, font_size=font_size, color=color, font=DEFAULT_FONT, weight=weight, **kwargs)


def make_embedding_bar(width: float = 1.0, height: float = 0.15, gradient: bool = True) -> VGroup:
    """Create a visual representation of an embedding vector."""
    bar = RoundedRectangle(
        width=width, height=height,
        corner_radius=height / 3,
        fill_opacity=0.6 if gradient else 0.4,
        stroke_width=1.5
    )
    if gradient:
        bar.set_fill(color=[GRUColors.embedding_gradient[0], GRUColors.embedding_gradient[1]])
    else:
        bar.set_fill(color=BLUE_C)
    bar.set_stroke(BLUE_D)

    outline = RoundedRectangle(
        width=width, height=height,
        corner_radius=height / 3,
        fill_opacity=0,
        stroke_width=1.5,
        stroke_color=WHITE
    )
    return VGroup(bar, outline)


def make_token_grid(rows: int = 4, cols: int = 4, cell_size: float = 0.4,
                    color=BLUE_C, show_indices: bool = False,
                    indices: list = None) -> VGroup:
    """Create a grid of token cells."""
    grid = VGroup()
    for i in range(rows):
        for j in range(cols):
            cell = Square(
                side_length=cell_size,
                color=color,
                fill_opacity=0.5,
                stroke_width=1.5
            )
            if show_indices and indices:
                idx = i * cols + j
                if idx < len(indices):
                    label = styled_text(str(indices[idx]), font_size=int(cell_size * 20), color=WHITE)
                    label.move_to(cell.get_center())
                    grid.add(VGroup(cell, label))
                else:
                    grid.add(cell)
            else:
                grid.add(cell)
    grid.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
    return grid


def make_gru_cell(width: float = 2.0, height: float = 1.2, label: str = "GRU") -> VGroup:
    """Create a visual representation of a GRU cell."""
    cell = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.15,
        color=PURPLE_A,
        fill_opacity=0.3,
        stroke_width=2
    )
    text = styled_text(label, font_size=18, color=WHITE, weight="BOLD")
    text.move_to(cell.get_center())
    return VGroup(cell, text)


def make_stacked_gru(num_layers: int = 3, width: float = 2.0,
                     height_per_layer: float = 0.8) -> VGroup:
    """Create a stacked GRU visualization."""
    layers = VGroup()
    for i in range(num_layers):
        layer = RoundedRectangle(
            width=width, height=height_per_layer,
            corner_radius=0.1,
            color=PURPLE_A,
            fill_opacity=0.3 + 0.1 * i,
            stroke_width=2
        )
        label = styled_text(f"GRU Layer {i+1}", font_size=14, color=WHITE)
        label.move_to(layer.get_center())
        layers.add(VGroup(layer, label))

    layers.arrange(UP, buff=0.1)
    return layers


def make_mlp_block(label: str = "MLP", width: float = 1.8, height: float = 0.8,
                   color=GREEN_C) -> VGroup:
    """Create an MLP block visualization."""
    block = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.1,
        color=color,
        fill_opacity=0.3,
        stroke_width=2
    )
    text = styled_text(label, font_size=14, color=WHITE)
    text.move_to(block.get_center())
    return VGroup(block, text)


def make_gaussian_distribution(width: float = 2.0, height: float = 1.0,
                               color=ORANGE) -> VGroup:
    """Create a visual representation of a Gaussian distribution."""
    # Bell curve using a polygon
    points = []
    num_points = 50
    for i in range(num_points + 1):
        x = -width/2 + (width * i / num_points)
        # Gaussian function
        y = height * np.exp(-((x * 2) ** 2) / 0.5)
        points.append([x, y, 0])

    # Close the shape at the bottom
    points.append([width/2, 0, 0])
    points.append([-width/2, 0, 0])

    curve = Polygon(*points, color=color, fill_opacity=0.4, stroke_width=2)

    # Add mu and sigma labels
    mu_label = MathTex(r"\mu", font_size=20, color=WHITE)
    mu_label.next_to(curve, DOWN, buff=0.1)

    return VGroup(curve, mu_label)


def make_state_vector(dim_label: str = "1024", width: float = 0.3,
                      height: float = 1.5, color=PURPLE_A) -> VGroup:
    """Create a visual representation of a state vector."""
    rect = Rectangle(
        width=width, height=height,
        color=color,
        fill_opacity=0.5,
        stroke_width=2
    )

    # Add dimension label
    label = styled_text(dim_label, font_size=12, color=WHITE)
    label.next_to(rect, DOWN, buff=0.1)

    return VGroup(rect, label)


def make_concat_visualization(vec1: VGroup, vec2: VGroup,
                              result_label: str = "2048") -> VGroup:
    """Create a visualization of vector concatenation."""
    # Position vectors side by side
    combined = VGroup(vec1.copy(), vec2.copy())
    combined.arrange(DOWN, buff=0)

    # Add bracket
    bracket_left = Line(
        combined.get_corner(UP + LEFT) + LEFT * 0.1,
        combined.get_corner(DOWN + LEFT) + LEFT * 0.1,
        color=WHITE
    )

    # Result label
    label = styled_text(result_label, font_size=12, color=WHITE)
    label.next_to(combined, DOWN, buff=0.15)

    return VGroup(combined, label)


def make_prediction_head(label: str = "Head", width: float = 1.8,
                         height: float = 0.6, color=GREEN_C) -> VGroup:
    """Create a prediction head block."""
    block = RoundedRectangle(
        width=width, height=height,
        corner_radius=0.1, color=color,
        fill_opacity=0.3, stroke_width=2
    )
    text = styled_text(label, font_size=14, color=WHITE)
    text.move_to(block.get_center())
    return VGroup(block, text)


def make_timestep_diagram(num_timesteps: int = 3, spacing: float = 3.0) -> VGroup:
    """Create a horizontal timeline with timestep markers."""
    diagram = VGroup()

    for t in range(num_timesteps):
        # Timestep marker
        marker = Circle(radius=0.15, color=WHITE, fill_opacity=0.3, stroke_width=2)
        label = styled_text(f"t={t}", font_size=12, color=WHITE)
        label.next_to(marker, DOWN, buff=0.1)

        marker_group = VGroup(marker, label)
        marker_group.move_to(RIGHT * t * spacing)
        diagram.add(marker_group)

        # Arrow to next timestep
        if t < num_timesteps - 1:
            arrow = Arrow(
                marker.get_right() + RIGHT * 0.2,
                marker.get_right() + RIGHT * (spacing - 0.4),
                color=GREY_B,
                stroke_width=2,
                tip_length=0.15
            )
            diagram.add(arrow)

    diagram.move_to(ORIGIN)
    return diagram
