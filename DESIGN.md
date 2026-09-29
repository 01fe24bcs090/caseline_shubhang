# CASELINE — Stitch Design System Specification (DESIGN.md)

> **Design Language**: Cyber-Forensic Dark / Precision Evidence Engine  
> **Target Audience**: Digital Forensics Investigators, Security Operations Center (SOC) Analysts, Law Enforcement  
> **Brand Mantra**: *"Evidence first. AI second. Human decision always."*

---

## 1. Visual Philosophy & Core Pillars

1. **Precision & Trust**: Digital evidence requires high legibility, strict timestamp formatting, cryptographic hashes (SHA-256), and zero visual ambiguity.
2. **Tactical Dark Mode**: Low-fatigue dark slate background (`#07090e`) engineered for multi-monitor operations centers with prolonged viewing sessions.
3. **Luminous Accents**: High-contrast cyan (`#00f0ff`), electric azure (`#38bdf8`), and alert states (Crimson `#f43f5e`, Amber `#f59e0b`, Emerald `#10b981`) designed to direct investigator focus to anomalies instantly.
4. **Glassmorphism & Depth**: Multi-layered elevation using translucent surfaces (`rgba(13, 18, 29, 0.75)`), 1px micro-borders, and subtle radial gradient glows.

---

## 2. Design Tokens

### 2.1 Color Palette

| Token Name | Hex / Value | Usage |
|---|---|---|
| `--color-bg-base` | `#07090e` | Deep background floor |
| `--color-surface-card` | `#0d121d` | Surface elevation 1 (Panels, Cards) |
| `--color-surface-hover` | `#131b2a` | Surface elevation 2 (Hover states, active pills) |
| `--color-surface-glass` | `rgba(13, 18, 29, 0.82)` | Glassmorphism with `backdrop-filter: blur(12px)` |
| `--color-border-subtle` | `rgba(56, 189, 248, 0.12)` | Container outlines and card dividers |
| `--color-border-bright` | `rgba(0, 240, 255, 0.38)` | Active borders, focus rings, hover glow |
| `--color-accent-cyan` | `#00f0ff` | Primary hero accent, key metric numbers |
| `--color-accent-azure` | `#38bdf8` | Secondary action buttons, interactive links |
| `--color-accent-indigo` | `#6366f1` | Feature highlights, AI pipeline badges |
| `--color-status-safe` | `#10b981` | NORMAL activity, verified badge swipes, granted access |
| `--color-status-warn` | `#f59e0b` | ATTENTION status, sensitive zone entry, dwell alerts |
| `--color-status-alert` | `#f43f5e` | SUSPICIOUS activity, ghost swipes, restricted zone breach |
| `--color-text-primary` | `#f8fafc` | Headings, vital evidence details, telemetry |
| `--color-text-secondary`| `#94a3b8` | Descriptions, labels, secondary metadata |
| `--color-text-muted` | `#64748b` | Timestamps, camera IDs, technical footnotes |

### 2.2 Typography

- **Display & Headings**: `'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
  - Weight: 600 (Semi-bold), 700 (Bold), 800 (Extra-bold)
  - Letter spacing: `-0.02em` for authoritative titles
- **Body & Controls**: `'Inter', system-ui, sans-serif`
  - Weight: 400 (Regular), 500 (Medium)
  - Line height: `1.55`
- **Monospace / Forensics Data**: `'JetBrains Mono', 'Fira Code', monospace`
  - Used for: Timestamps, SHA-256 hashes, Track IDs, Coordinates `(x, y, w, h)`, Bounding Boxes, Confidence scores `0.942`

---

## 3. Component Specifications

### 3.1 Forensic Header Banner (SOC HUD)
- **Container**: Full-width glassmorphic banner with top border gradient `linear-gradient(90deg, #00f0ff, #6366f1, #00f0ff)`.
- **System Indicator**: Pulsing green beacon (`● SYSTEM OPERATIONAL`) next to case ID.
- **Pills**: Compact mono pills displaying SHA-256 integrity, active detector (`YOLO11n`), and ByteTrack status.

### 3.2 Evidence Cards & Metric Chips
- **Background**: `rgba(13, 18, 29, 0.75)` with `1px solid rgba(56, 189, 248, 0.14)`.
- **Corner Radius**: `10px`.
- **Hover State**: Elevates by `-2px` with luminous border `rgba(0, 240, 255, 0.4)` and box shadow `0 8px 24px -4px rgba(0, 240, 255, 0.12)`.

### 3.3 Navigation Tabs
- **Layout**: Clean segmented tab bar with zero default browser borders.
- **Active Tab Indicator**: Luminous cyan bottom line (`height: 2px; box-shadow: 0 0 12px #00f0ff`) with text highlighted in `#f8fafc`.
- **Inactive Tab**: `#94a3b8` text with smooth hover color transition (`#38bdf8`).

### 3.4 Buttons & Interactive Controls
- **Primary Action**: Gradient `linear-gradient(135deg, #0284c7 0%, #00f0ff 100%)`, bold white text, subtle text shadow, rounded `8px`, glowing box-shadow on hover.
- **Danger Action**: Gradient `linear-gradient(135deg, #be123c 0%, #f43f5e 100%)`.
- **Form Inputs**: Dark slate backgrounds (`#0a0f19`), inset subtle shadow, cyan focus halo.

### 3.5 Status Badges & Pill Tags
- `NORMAL` / `VERIFIED`: Emerald pill with translucent background `rgba(16, 185, 129, 0.15)` and border `rgba(16, 185, 129, 0.4)`.
- `ATTENTION`: Amber pill with `rgba(245, 158, 11, 0.15)` and border `rgba(245, 158, 11, 0.4)`.
- `SUSPICIOUS` / `VIOLATION`: Crimson pill with `rgba(244, 63, 94, 0.15)` and border `rgba(244, 63, 94, 0.4)`.

---

## 4. Stitch Integration Guide

When using **Google Stitch** (`stitch.withgoogle.com`):
1. Import this `DESIGN.md` to feed Stitch the design tokens and layout constraints.
2. Stitch will automatically align all AI-generated screens, components, and code with CaseLine's cyber-forensic design language.
3. Generated HTML/CSS or React components can be transferred directly into the CaseLine application interface.
