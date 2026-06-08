---
name: AgrypniaOxus
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#e7bcbc'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#ae8787'
  outline-variant: '#5e3f3f'
  surface-tint: '#ffb3b3'
  primary: '#ffb3b3'
  on-primary: '#680015'
  primary-container: '#ff5261'
  on-primary-container: '#5b0011'
  inverse-primary: '#bf002f'
  secondary: '#c6c6c7'
  on-secondary: '#2f3131'
  secondary-container: '#454747'
  on-secondary-container: '#b4b5b5'
  tertiary: '#c6c6c6'
  on-tertiary: '#2f3131'
  tertiary-container: '#909191'
  on-tertiary-container: '#282a2a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdad9'
  primary-fixed-dim: '#ffb3b3'
  on-primary-fixed: '#400009'
  on-primary-fixed-variant: '#920022'
  secondary-fixed: '#e2e2e2'
  secondary-fixed-dim: '#c6c6c7'
  on-secondary-fixed: '#1a1c1c'
  on-secondary-fixed-variant: '#454747'
  tertiary-fixed: '#e3e2e2'
  tertiary-fixed-dim: '#c6c6c6'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#464747'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
  signal-red: '#fc1c46'
  pure-white: '#ffffff'
  ash: '#cccccc'
  graphite: '#4c4c4c'
  void: '#000000'
  obsidian-canvas: '#0a0a0a'
  carbon-panel: '#1a1a1a'
  smoke-overlay: '#2a2a2a'
typography:
  display:
    fontFamily: Hanken Grotesk
    fontSize: 198px
    fontWeight: '300'
    lineHeight: '0.92'
    letterSpacing: -0.067em
  headline-lg:
    fontFamily: Hanken Grotesk
    fontSize: 91px
    fontWeight: '300'
    lineHeight: '0.96'
    letterSpacing: -0.04em
  headline-lg-mobile:
    fontFamily: Hanken Grotesk
    fontSize: 48px
    fontWeight: '300'
    lineHeight: '1.0'
    letterSpacing: -0.02em
  heading:
    fontFamily: Hanken Grotesk
    fontSize: 72px
    fontWeight: '300'
    lineHeight: '1.0'
    letterSpacing: -0.02em
  heading-sm:
    fontFamily: Hanken Grotesk
    fontSize: 27px
    fontWeight: '400'
    lineHeight: '1.15'
    letterSpacing: -0.009em
  subheading:
    fontFamily: Hanken Grotesk
    fontSize: 18px
    fontWeight: '500'
    lineHeight: '1.2'
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 17px
    fontWeight: '400'
    lineHeight: '1.5'
  body-sm:
    fontFamily: Hanken Grotesk
    fontSize: 15px
    fontWeight: '400'
    lineHeight: '1.5'
  caption:
    fontFamily: Hanken Grotesk
    fontSize: 14px
    fontWeight: '500'
    lineHeight: '1.2'
  micro:
    fontFamily: Hanken Grotesk
    fontSize: 10px
    fontWeight: '400'
    lineHeight: '2.14'
spacing:
  element-gap: 9px
  card-padding: 22px
  section-gap-min: 86px
  section-gap-max: 108px
  container-padding-h: 126px
  margin-top-subtext: 29px
---

## Brand & Style

The brand operates at the intersection of cinematic tension and architectural precision. The identity is defined by a "crimson flare in obsidian dark," evoking an emotional response of focused intensity and sophisticated silence. The aesthetic is purely editorial, stripping away all non-essential ornamentation to let high-contrast typography and a singular, vibrant signal color command the visual field.

The design style is a hybrid of **Minimalism** and **High-Contrast / Bold**, utilizing vast negative space and "weightless" oversized display elements. Every interaction is intended to feel deliberate and surgical, set against a void that prioritizes content legibility and brand authority.

## Colors

The palette is anchored in a deep obsidian dark mode. The chromatic strategy is strictly functional: **Signal Red** is the sole carrier of action and brand identity. 

- **Primary (Signal Red):** Reserved for primary action buttons, active states, and brand marks. Only one red element should occupy a viewport at any given time.
- **Secondary (Pure White):** Used for high-impact display headlines and primary labels to ensure maximum contrast against the dark canvas.
- **Tertiary (Ash):** The workhorse for body text and secondary information, providing readability without competing with the primary headlines.
- **Neutral (Obsidian):** Three tiers of depth are used to separate surfaces—Obsidian Canvas for the background, Carbon Panel for cards, and Smoke Overlay for modals.

## Typography

This design system utilizes a single typeface—**Hanken Grotesk**—to maintain architectural unity. The signature of the system is the interplay between massive scale and light weight.

- **Display Headlines:** Set at 198px with weight 300. Use extreme negative tracking (-0.067em) to transform words into sculptural shapes. These should be left-aligned and allowed to break naturally.
- **Body Copy:** Kept compact (15-17px) but given generous leading (1.5) to ensure breathability on dark surfaces.
- **Micro-labels:** Used for utility elements like "(SCROLL)" or metadata, often bracketed or all-caps.
- **Mobile scaling:** On small viewports, the display headlines must scale down aggressively (e.g., 48px-72px) while maintaining the light weight and negative tracking.

## Layout & Spacing

The layout is a **fluid grid** that prioritizes negative space as a load-bearing structural element. 

- **Grid:** A 12-column system is used loosely, but the core philosophy is editorial. Large headlines often span 8-10 columns, while body copy is constrained to a max-width of ~520px (approx. 4-6 columns) to maintain readability.
- **Margins:** Desktop layouts utilize a generous 126px horizontal gutter.
- **Vertical Rhythm:** Major sections are separated by substantial 86px to 108px gaps to maintain the "weightless void" atmosphere.
- **Breakpoints:**
    - **Desktop:** Full-bleed canvas, 126px margins, left-aligned headlines.
    - **Tablet:** Margins reduce to 64px, section gaps reduce to 72px.
    - **Mobile:** Margins reduce to 24px, display typography scales down to headline-lg-mobile, single-column reflow.

## Elevation & Depth

Visual hierarchy is achieved through **tonal layers** and spatial tension rather than shadows. 

- **Surfaces:** Depth is indicated by three dark tiers: `#0a0a0a` (base), `#1a1a1a` (panels/cards), and `#2a2a2a` (overlays). 
- **Shadows:** No ambient or drop shadows are permitted. The UI is intentionally flat.
- **Dimensionality:** A single iridescent 3D orb is used as the hero visual. Depth is suggested by overlapping oversized text across this 3D object.
- **Dividers:** Hairline borders (1px) in Graphite or Ash are used for clean section breaks without adding visual weight.

## Shapes

The system follows a binary shape language: **Sharp or Pill-shaped**.

- **Structure:** All containers, cards, input fields, and panels use 0px border-radius (Sharp). This reinforces the architectural and brutalist-minimalist feel.
- **Interaction:** Pill-shaped elements (rounded-full / 9999px) are reserved exclusively for interactive elements like buttons and chips. This creates a clear visual distinction between static content and actionable triggers.

## Components

### Buttons
- **Signal CTA:** Pill-shaped, #fc1c46 background, #ffffff text (weight 500). Padding: 9px vertical, 22px horizontal. No border or shadow.
- **Secondary CTA:** Pill-shaped, 1px Pure White border, #ffffff text, transparent background.

### Cards
- **Structure:** 0px radius, Carbon Panel (#1a1a1a) background.
- **Padding:** Uniform 22px on all sides.
- **Interactions:** Hover states should use a subtle tonal shift to Smoke Overlay (#2a2a2a).

### Input Fields
- **Styling:** 0px radius, 1px Graphite (#4c4c4c) bottom border only or full hairline border.
- **Typography:** Ash (#cccccc) for placeholder text, Pure White (#ffffff) for active input.

### Navigation
- **Header:** Full-width row with 126px padding. Left: Wordmark. Center: Centered tagline bar. Right: Primary CTA and Ghost Menu Trigger.
- **Menu Trigger:** Minimal 2px stroke icon in Pure White, no background.

### Lists & Dividers
- **Dividers:** 1px solid Graphite (#4c4c4c).
- **List Items:** Ash text with Signal Red markers or active states.

### Interactive Feedback
- Use Signal Red (#fc1c46) exclusively for active states, hover accents, and brand markers to ensure tap targets are unmistakable in the void.