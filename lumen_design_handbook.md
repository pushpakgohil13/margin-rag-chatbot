# Lumen Design System — Team Handbook

This handbook explains how the product design team builds, reviews and ships work using the Lumen design system. It is a fictional sample document used to demonstrate the Margin RAG chatbot.

# About Lumen

Lumen is the shared design system used by all product squads. It contains foundations (color, type, spacing, motion), a component library in Figma, and matching React components maintained by the front-end platform team. The design system team has four people: one lead designer, two product designers and one design engineer. Lumen releases a new version on the first Monday of every month.

# Color tokens

Never use raw hex values in a design file. Always use a semantic color token such as color.text.primary or color.surface.raised. Brand colors are only used for marketing surfaces and must not appear inside the product UI. Every color token has a light-mode and a dark-mode value, and designers must check both modes before handoff.

# Accessibility standards

All product screens must meet WCAG 2.2 level AA. Body text needs a contrast ratio of at least 4.5:1 against its background. Large text (24px and above, or 18.5px bold) and UI icons need at least 3:1. Interactive targets must be at least 44 by 44 pixels on touch devices. Every flow must be fully usable with a keyboard, and focus states must always be visible. Color must never be the only way information is shown; pair it with an icon, label or pattern.

# Typography

The product typeface is Inter for UI text and JetBrains Mono for code. The type scale has seven steps: 12, 14, 16, 20, 24, 32 and 40 pixels. Body text is 16px with a line height of 1.5. Do not use more than three type sizes on a single screen.

# Spacing and layout

Lumen uses a 4px base spacing unit. All padding and margins must be multiples of 4px, with the most common values being 8, 16, 24 and 32. Layouts use a 12-column grid on desktop with 24px gutters, an 8-column grid on tablet and a 4-column grid on mobile with 16px gutters. Maximum content width is 1280px.

# Icons

Icons come in three sizes: 16px, 20px and 24px. Use the 20px size by default. All icons are outlined with a 1.5px stroke. Custom icons must be submitted to the design system team for review before they are used in production.

# Design critique

Design crits happen every Wednesday from 3:00 to 4:00 pm. Anyone can bring work at any fidelity. Book a slot in the #design-crit channel by Tuesday at 6 pm. Each slot is 20 minutes: 5 minutes of context, 10 minutes of feedback and 5 minutes to agree on next steps. Feedback should be framed around the user problem and the brief, not personal taste.

# Requesting a new component

If no existing component fits, fill in the component request form in Notion. Include the problem, at least two real screens where it will be used, and a link to any research. The design system team reviews requests every Friday. The standard turnaround for a new component, from approval to release in Figma and code, is 10 business days. Urgent requests need sign-off from the head of design.

# Figma file conventions

Name files using the pattern Squad / Feature / Stage, for example Payments / Refund flow / Explorations. Every file must have a cover page with the owner, status and last-updated date. Status labels are Exploring, In review, Ready for dev and Shipped. Detach components only as a last resort, and leave a comment explaining why.

# Handoff to engineering

A design is ready for development only when it passes the handoff checklist: all states are designed (empty, loading, error, success), both light and dark modes are checked, copy has been reviewed by the content designer, accessibility annotations are added, and the file status is set to Ready for dev. Designers attend the sprint kickoff for any feature they hand off.

# Deprecating components

When a component is replaced, it is marked as deprecated in Figma and code. Deprecated components are kept for two monthly releases and then removed. Squads get a migration guide and a reminder in the #lumen-updates channel one week before removal.

# Research and testing

Every major feature needs at least five usability test sessions before release. Research notes and recordings are stored in the Research Hub in Dovetail. Participants are paid a 50 dollar voucher for a 45 minute session. Personal data from research must be deleted after 12 months.

# Tools and access

The team uses Figma for design, Notion for documentation, Dovetail for research and Linear for tracking work. New designers get Figma editor access on their first day and Dovetail access after completing the research privacy training.
