"""Werewolf game layer: roles, factions, and the benchmark's default configuration.

The engine core stays game-agnostic (a player's `role` is an opaque string); this
package pins the concrete Werewolf role set, the role->faction map every win check
reads, and the 7-player default config the benchmark rates games on.
"""
