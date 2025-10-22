"""Shared helper utilities to give our world-model scenes a 3Blue1Brown feel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from manim import (
    VGroup,
    Text,
    Rectangle,
    RoundedRectangle,
    Arrow,
    CurvedArrow,
    FadeIn,
    FadeOut,
    LaggedStart,
    LaggedStartMap,
    Indicate,
    Integer,
    DecimalNumber,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    BLUE_E,
    BLUE_C,
    BLUE_B,
    TEAL_D,
    PURPLE_A,
    ORANGE,
    YELLOW,
    WHITE,
)


@dataclass(frozen=True)
class ThreeBPalette:
    """Central place to tweak colours so the scenes stay consistent."""

    token_fill: str = BLUE_C
    action_fill: str = TEAL_D
    query_fill: str = PURPLE_A
    caption: str = WHITE
    attention_self: str = YELLOW
    attention_cross: str = ORANGE


def make_token_strip(num_cells: int, cell_size: float = 0.4, fill_color: str | None = None) -> VGroup:
    """Return a horizontal strip of equally-sized rectangles."""

    palette = ThreeBPalette()
    color = fill_color or palette.token_fill

    cells = VGroup(
        Rectangle(width=cell_size, height=cell_size, color=WHITE, fill_opacity=0.35)
        for _ in range(num_cells)
    )
    for cell in cells:
        cell.set_fill(color, opacity=0.4)
    cells.arrange(RIGHT, buff=0.07)
    return cells


def add_caption(base_mobject, text: str, font_size: int = 24, buff: float = 0.3, above: bool = True) -> Text:
    caption = Text(text, font_size=font_size, color=ThreeBPalette().caption)
    direction = UP if above else DOWN
    caption.next_to(base_mobject, direction, buff=buff)
    caption.set_x(base_mobject.get_center()[0])
    return caption


def animate_positional_overlay(scene, targets: Sequence, highlight_color=ORANGE, run_time: float = 1.3):
    """Apply a brief highlight animation to a sequence of rectangles or dots."""

    scene.play(
        LaggedStartMap(
            Indicate,
            VGroup(*targets),
            color=highlight_color,
            scale_factor=1.04,
            lag_ratio=0.06,
            run_time=run_time,
        )
    )


def attention_flow(scene, sources: Sequence, targets: Sequence, weights: Iterable[float], *,
                   color: str = ORANGE, run_time: float = 2.0, linger: float = 2.0,
                   curve: float = 0.4):
    """Animate a set of arrows whose stroke width corresponds to attention weights."""

    arrows = VGroup()
    for src, dst, weight in zip(sources, targets, weights):
        src_point = src.get_center()
        dst_point = dst.get_center()
        arrow = CurvedArrow(
            src_point,
            dst_point,
            angle=curve,
            color=color,
            stroke_width=6 * weight,
            tip_length=0.18,
        )
        arrows.add(arrow)

    if len(arrows) == 0:
        return

    scene.play(LaggedStart(*[FadeIn(arrow) for arrow in arrows], lag_ratio=0.15, run_time=run_time))
    scene.wait(linger)
    scene.play(FadeOut(arrows))


def make_numeric_embedding(width: float = 3.5, height: float = 0.45, columns: int = 10,
                           color_gradient: tuple[str, str] = (BLUE_E, BLUE_B)) -> VGroup:
    """Simple horizontal bar of cells coloured with a gradient; used for embeddings."""

    left_color, right_color = color_gradient
    cells = VGroup()
    for idx in range(columns):
        rect = Rectangle(width=width / columns, height=height, stroke_width=0)
        alpha = idx / max(columns - 1, 1)
        rect.set_fill(interpolate_color(left_color, right_color, alpha), opacity=0.9)
        cells.add(rect)
    cells.arrange(RIGHT, buff=0)
    frame = RoundedRectangle(width=width, height=height + 0.08, corner_radius=0.08, color=WHITE)
    embedding = VGroup(frame, cells)
    cells.move_to(frame)
    return embedding


def add_legend(scene, entries: Sequence[tuple[str, str]], base, buff: float = 0.2, horizontal: bool = True) -> VGroup:
    """Create a legend with coloured swatches to explain strips (tokens, actions, etc.)."""

    items = VGroup()
    for label, color in entries:
        swatch = Rectangle(width=0.3, height=0.3, color=WHITE, fill_opacity=0.35)
        swatch.set_fill(color, opacity=0.7)
        text = Text(label, font_size=20, color=ThreeBPalette().caption)
        text.next_to(swatch, RIGHT, buff=0.15)
        item = VGroup(swatch, text)
        items.add(item)

    direction = RIGHT if horizontal else DOWN
    items.arrange(direction, buff=0.4, aligned_edge=LEFT)
    items.next_to(base, DOWN, buff=buff)
    max_x = max(item.get_x(RIGHT) for item in items)
    min_x = min(item.get_x(LEFT) for item in items)
    items.shift(-((max_x + min_x) / 2) * RIGHT)
    items.shift(base.get_center()[0] * RIGHT)
    return items


# Utilities below need functions from manim.mobject import but keep explicit to avoid manim_imports_ext.
from manim.utils.color import interpolate_color


def next_token_bar_chart(words: Sequence[str], probs: Sequence[float], width_100p: float = 2.8,
                         bar_height: float = 0.25, font_size: int = 24,
                         prob_power: float = 0.75, use_percent: bool = True) -> VGroup:
    """Return a horizontal bar chart similar to 3b1b autogression scenes."""

    bars = VGroup()
    labels = VGroup()
    prob_labels = VGroup()

    for word, prob in zip(words, probs):
        bar_width = (prob ** prob_power) * width_100p
        bar = Rectangle(width=bar_width, height=bar_height, stroke_color=WHITE, stroke_width=1)
        bar.set_fill(TEAL_D, opacity=0.9)
        bars.add(bar)

        label = Text(word, font_size=font_size, color=ThreeBPalette().caption)
        label.next_to(bar, LEFT, buff=0.15)
        labels.add(label)

        if use_percent:
            value = Integer(int(round(prob * 100)), unit="%", font_size=int(font_size * 0.75))
        else:
            value = DecimalNumber(prob, num_decimal_places=2, font_size=int(font_size * 0.75))
        value.next_to(bar, RIGHT, buff=0.15)
        prob_labels.add(value)

    rows = VGroup(*(VGroup(label, bar, prob) for label, bar, prob in zip(labels, bars, prob_labels)))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25 * bar_height)

    base_left = min(group[1].get_x(LEFT) for group in rows)
    for group in rows:
        shift = -base_left - 0.35
        group.shift(shift * RIGHT)

    return rows
