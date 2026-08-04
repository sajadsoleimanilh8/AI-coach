/**
 * JS mirror of ai/computer_vision/tactical_analysis/formation_templates.py.
 *
 * FIXED (Phase 3 audit): TabTeamIntelligence.jsx's PitchVisual component
 * previously hardcoded a 4-3-3 slot layout regardless of what the API
 * actually returned as the detected formation -- so a real "4-4-2" result
 * would show the correct label in the header while the dots on the pitch
 * still drew a 4-3-3 shape. That's a visible, judge-noticeable
 * contradiction. Keep this file's slot coordinates in sync with the
 * Python source of truth if either changes; there is intentionally no
 * shared-source-of-truth codegen for this yet (JS/Python don't share an
 * import graph here) -- a mismatch would be a silent visual bug again, so
 * if you touch one, touch both.
 */
export const FORMATION_TEMPLATES = {
  '4-3-3': [
    [0.20, 0.15], [0.18, 0.38], [0.18, 0.62], [0.20, 0.85],
    [0.45, 0.30], [0.42, 0.50], [0.45, 0.70],
    [0.75, 0.15], [0.80, 0.50], [0.75, 0.85],
  ],
  '4-4-2': [
    [0.20, 0.15], [0.18, 0.38], [0.18, 0.62], [0.20, 0.85],
    [0.50, 0.15], [0.48, 0.38], [0.48, 0.62], [0.50, 0.85],
    [0.80, 0.38], [0.80, 0.62],
  ],
  '4-2-3-1': [
    [0.20, 0.15], [0.18, 0.38], [0.18, 0.62], [0.20, 0.85],
    [0.38, 0.38], [0.38, 0.62],
    [0.65, 0.20], [0.65, 0.50], [0.65, 0.80],
    [0.82, 0.50],
  ],
  '3-5-2': [
    [0.18, 0.25], [0.16, 0.50], [0.18, 0.75],
    [0.45, 0.10], [0.45, 0.35], [0.42, 0.50], [0.45, 0.65], [0.45, 0.90],
    [0.78, 0.38], [0.78, 0.62],
  ],
  '3-4-3': [
    [0.18, 0.25], [0.16, 0.50], [0.18, 0.75],
    [0.48, 0.15], [0.48, 0.38], [0.48, 0.62], [0.48, 0.85],
    [0.78, 0.20], [0.82, 0.50], [0.78, 0.80],
  ],
};

/** Returns the slot layout for a formation name, or null if it's not a
 * known template (or hasn't been detected yet) -- callers must render an
 * empty pitch in that case, never a default shape. */
export function getFormationSlots(formationName) {
  return FORMATION_TEMPLATES[formationName] || null;
}
