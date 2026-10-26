"""
Pip, Margin's mascot: a glossy gradient orb with a living colour swirl and simple white eyes.

Moods: greet, idle, reading, thinking, talking, happy, confused.
"""

import base64
import html as _html

MOODS = ("greet", "idle", "reading", "thinking", "talking", "happy", "confused")


EXTRAS = (
    '<g class="pip-happy" stroke="#fff" stroke-width="5" fill="none" stroke-linecap="round">'
    '<path d="M55 70 Q63 58 71 70"/><path d="M87 70 Q95 58 103 70"/></g>'
    '<rect class="pip-squint" x="86" y="66" width="18" height="6" rx="3" fill="#fff"/>'
    '<ellipse class="pip-mouth" cx="79" cy="98" rx="6" ry="4.5" fill="#fff" opacity=".92"/>'
    '<g class="pip-dots" fill="#fff"><circle cx="60" cy="104" r="4"/><circle cx="79" cy="104" r="4"/>'
    '<circle cx="98" cy="104" r="4"/></g>'
    '<g class="pip-page"><rect x="104" y="92" width="34" height="42" rx="6" fill="#fff"/>'
    '<g stroke="#C4B5FD" stroke-width="3" stroke-linecap="round"><path d="M111 104 H131"/>'
    '<path d="M111 112 H127"/><path d="M111 120 H130"/></g></g>'
    '<text class="pip-q" x="122" y="44" font-size="26" font-weight="700" fill="#A78BFA" font-family="sans-serif">?</text>'
    '<g class="pip-spark" fill="#FBBF24"><path d="M24 34 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3z"/>'
    '<path d="M136 104 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z"/></g>'
)


def _svg(uid="p", size_class="pip-svg", extras=True):
    """The orb. uid keeps gradient ids unique when several orbs are on one page."""
    head = (
        f'<svg class="{size_class}" viewBox="0 0 160 170" role="img" aria-label="Pip, the Margin mascot">'
        "<defs>"
        f'<clipPath id="{uid}c"><circle cx="80" cy="80" r="60"/></clipPath>'
        f'<filter id="{uid}b" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="13"/></filter>'
        f'<filter id="{uid}g" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="9"/></filter>'
        f'<radialGradient id="{uid}s" cx="34%" cy="26%" r="70%">'
        '<stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset=".35" stop-color="#fff" stop-opacity=".08"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
        f'<radialGradient id="{uid}r" cx="50%" cy="50%" r="50%">'
        '<stop offset=".82" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff" stop-opacity=".55"/></radialGradient>'
        "</defs>"
        f'<ellipse class="pip-glow" cx="80" cy="152" rx="46" ry="9" fill="#F472B6" opacity=".38" filter="url(#{uid}g)"/>'
        '<g class="pip-bob"><g class="pip-tilt">'
        f'<g clip-path="url(#{uid}c)">'
        '<rect x="0" y="0" width="160" height="160" fill="#B5177F"/>'
        f'<g class="pip-swirl" filter="url(#{uid}b)">'
        '<circle cx="48" cy="44" r="40" fill="#FF8AD6"/>'
        '<circle cx="122" cy="72" r="42" fill="#7C5CFF"/>'
        '<circle cx="58" cy="124" r="36" fill="#FF7A2F"/>'
        '<circle cx="110" cy="126" r="28" fill="#E879F9"/>'
        '<circle cx="84" cy="86" r="22" fill="#A0136F"/>'
        "</g>"
        f'<circle cx="80" cy="80" r="60" fill="url(#{uid}s)"/>'
        f'<circle cx="80" cy="80" r="60" fill="url(#{uid}r)"/>'
        "</g>"
        '<g class="pip-eyes" fill="#fff">'
        '<rect x="54" y="51" width="16" height="30" rx="8"/><rect x="88" y="51" width="16" height="30" rx="8"/>'
        "</g>"
    )
    return head + (EXTRAS if extras else "") + "</g></g></svg>"


CSS = """
.pip-svg{display:block;overflow:visible}
.pip-svg *{transform-box:fill-box;transform-origin:center}
.pip-swirl{transform-box:view-box!important;transform-origin:80px 80px!important;animation:pipSpin 14s linear infinite}
.pip-happy,.pip-squint,.pip-mouth,.pip-dots,.pip-page,.pip-q,.pip-spark{opacity:0}
.pip-bob{animation:pipFloat 3.6s ease-in-out infinite}
.pip-glow{animation:pipGlowPulse 3.6s ease-in-out infinite}
.pip-eyes{animation:pipBlink 4.8s infinite}
[data-mood=greet] .pip-bob{animation:pipHop 1.6s ease-in-out infinite}
[data-mood=reading] .pip-eyes{animation:pipScan 1.6s ease-in-out infinite}
[data-mood=reading] .pip-page{opacity:1;animation:pipPage 2.2s ease-in-out infinite}
[data-mood=reading] .pip-swirl,[data-mood=thinking] .pip-swirl{animation-duration:4s}
[data-mood=thinking] .pip-eyes{transform:translate(-5px,-6px) scaleY(.85)}
[data-mood=thinking] .pip-dots{opacity:1}
[data-mood=thinking] .pip-dots circle{animation:pipDot 1.2s ease-in-out infinite}
[data-mood=thinking] .pip-dots circle:nth-child(2){animation-delay:.2s}
[data-mood=thinking] .pip-dots circle:nth-child(3){animation-delay:.4s}
[data-mood=talking] .pip-mouth{opacity:.92;animation:pipTalk .3s ease-in-out infinite alternate}
[data-mood=talking] .pip-swirl{animation-duration:7s}
[data-mood=happy] .pip-eyes{opacity:0;animation:none}
[data-mood=happy] .pip-happy,[data-mood=happy] .pip-spark{opacity:1}
[data-mood=happy] .pip-bob{animation:pipBounce .9s cubic-bezier(.3,1.6,.5,1) 2,pipFloat 3.6s ease-in-out 1.8s infinite}
[data-mood=happy] .pip-spark path{animation:pipTwinkle 1.4s ease-in-out infinite}
[data-mood=happy] .pip-spark path:nth-child(2){animation-delay:.5s}
[data-mood=confused] .pip-tilt{transform:rotate(-10deg)}
[data-mood=confused] .pip-eyes rect:nth-child(2){opacity:0}
[data-mood=confused] .pip-squint,[data-mood=confused] .pip-q{opacity:1}
[data-mood=confused] .pip-swirl{animation-duration:30s}
@keyframes pipSpin{to{transform:rotate(360deg)}}
@keyframes pipFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes pipHop{0%,100%{transform:translateY(0)}30%{transform:translateY(-10px)}60%{transform:translateY(0) scale(1.03,.97)}}
@keyframes pipGlowPulse{0%,100%{opacity:.38}50%{opacity:.22}}
@keyframes pipBounce{0%{transform:translateY(0)}35%{transform:translateY(-16px) scale(1.03,.97)}70%{transform:translateY(0) scale(.98,1.02)}100%{transform:translateY(0)}}
@keyframes pipBlink{0%,92%,100%{transform:scaleY(1)}95%{transform:scaleY(.1)}}
@keyframes pipScan{0%,100%{transform:translateX(-5px)}50%{transform:translateX(5px)}}
@keyframes pipPage{0%,100%{transform:translateY(0) rotate(6deg)}50%{transform:translateY(-3px) rotate(10deg)}}
@keyframes pipDot{0%,100%{opacity:.3;transform:scale(.75)}50%{opacity:1;transform:scale(1)}}
@keyframes pipTalk{from{transform:scaleY(.4)}to{transform:scaleY(1.15)}}
@keyframes pipTwinkle{0%,100%{opacity:.2;transform:scale(.6)}50%{opacity:1;transform:scale(1.1)}}
@keyframes pipPop{from{opacity:0;transform:translateY(6px) scale(.96)}to{opacity:1;transform:none}}
@keyframes pipFade{to{opacity:0;transform:translateY(4px)}}
.pip-hero{display:flex;flex-direction:column;align-items:center;text-align:center;margin:.4rem 0 .2rem 0;position:relative;padding-top:44px}
.pip-hero .pip-svg{width:150px;height:160px}
.pip-hero .pip-bubble{position:absolute;top:0;right:calc(50% + 34px);margin:0;white-space:nowrap}
.pip-hero .pip-bubble:after{content:"";position:absolute;right:16px;bottom:-6px;width:12px;height:12px;background:#fff;
  transform:rotate(45deg);box-shadow:3px 3px 6px rgba(76,52,140,.06)}
.pip-dock{position:fixed;right:28px;bottom:118px;z-index:999;display:flex;align-items:flex-end;gap:8px;pointer-events:none;
  flex-direction:row-reverse}
.pip-dock .pip-svg{width:74px;height:78px}
.pip-bubble{background:#fff;color:#1F1B2D;border-radius:18px;padding:.55rem .9rem;font-size:.9rem;line-height:1.45;max-width:260px;
  box-shadow:0 10px 30px rgba(76,52,140,.14),0 1px 2px rgba(76,52,140,.08);border:1px solid #EFEAFB;margin-bottom:34px;
  position:relative;animation:pipPop .35s cubic-bezier(.2,1.4,.4,1) both;text-align:left}
.pip-bubble b{display:block;font-size:.72rem;font-weight:600;color:#9B7BF7;margin-bottom:.1rem}
.pip-dock .pip-bubble:after{content:"";position:absolute;right:-6px;bottom:14px;width:12px;height:12px;background:#fff;
  border-right:1px solid #EFEAFB;border-top:1px solid #EFEAFB;transform:rotate(45deg)}
.pip-dock.pip-autohide .pip-bubble{animation:pipPop .35s cubic-bezier(.2,1.4,.4,1) both,pipFade .6s ease 8s forwards}
@media (max-width:700px){.pip-dock{right:12px;bottom:112px}.pip-dock .pip-svg{width:54px;height:58px}
  .pip-dock .pip-bubble{max-width:180px;font-size:.8rem;margin-bottom:24px}
  .pip-hero .pip-bubble{position:static;margin-bottom:.4rem}.pip-hero .pip-bubble:after{display:none}}
@media (prefers-reduced-motion:reduce){.pip-svg *,.pip-bubble{animation:none!important}}
"""


def render(mood, message, mode="dock", autohide=False, uid="pip"):
    mood = mood if mood in MOODS else "idle"
    cls = ("pip-dock" + (" pip-autohide" if autohide else "")) if mode == "dock" else "pip-hero"
    bubble = f'<div class="pip-bubble"><b>Pip</b>{_html.escape(message)}</div>' if message else ""
    return f'<div class="{cls}" data-mood="{mood}">{_svg(uid)}{bubble}</div>'


def icon(mood="idle", size=28, uid="pi"):
    """Small static orb for avatars and the logo."""
    return f'<span class="pip-icon" data-mood="{mood}" style="display:inline-block;width:{size}px;height:{size}px">{_svg(uid)}</span>'


def avatar_data_uri():
    """A static orb as an SVG data URI, used as the chat avatar."""
    svg = _svg("av", extras=False).replace('class="pip-svg"', 'xmlns="http://www.w3.org/2000/svg"')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
