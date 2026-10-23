"""
Pip, Margin's mascot. An animated SVG character with seven moods, plus a speech bubble.

    greet     waving hello
    idle      floating and blinking
    reading   eyes scanning a page it holds
    thinking  looking up, thought dots
    talking   mouth moving while the answer streams
    happy     smiling eyes, bounce, sparkles
    confused  tilted, sweat drop (answer not found or an error)
"""

import html as _html

MOODS = ("greet", "idle", "reading", "thinking", "talking", "happy", "confused")

CSS = """
.pip-dock{position:fixed;right:22px;bottom:70px;z-index:1000;display:flex;align-items:flex-end;gap:10px;pointer-events:none;flex-direction:row-reverse}
.pip-hero{display:flex;align-items:center;gap:22px;margin:.4rem 0 1.4rem 0}
.pip-svg{display:block;overflow:visible}
.pip-dock .pip-svg{width:92px;height:92px;filter:drop-shadow(0 10px 22px rgba(124,92,255,.35))}
.pip-hero .pip-svg{width:150px;height:150px;flex:none;filter:drop-shadow(0 16px 32px rgba(124,92,255,.40))}
.pip-bubble{position:relative;background:rgba(23,28,54,.86);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,.10);color:#E7E9F3;border-radius:16px;padding:.65rem .9rem;font-size:.9rem;line-height:1.45;
  max-width:270px;box-shadow:0 12px 30px rgba(0,0,0,.25);margin-bottom:46px;animation:pipPop .35s cubic-bezier(.2,1.4,.4,1) both}
.pip-hero .pip-bubble{max-width:440px;font-size:1rem;margin-bottom:0}
.pip-bubble b.pip-name{display:block;font-size:.72rem;font-weight:600;letter-spacing:.02em;
  background:linear-gradient(90deg,#A78BFA,#38BDF8);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:.15rem}
.pip-dock .pip-bubble:after{content:"";position:absolute;right:-7px;bottom:16px;width:14px;height:14px;background:inherit;
  border-right:1px solid rgba(255,255,255,.10);border-top:1px solid rgba(255,255,255,.10);transform:rotate(45deg)}
.pip-hero .pip-bubble:after{content:"";position:absolute;left:-7px;top:50%;margin-top:-7px;width:14px;height:14px;background:inherit;
  border-left:1px solid rgba(255,255,255,.10);border-bottom:1px solid rgba(255,255,255,.10);transform:rotate(45deg)}
.pip-dock.pip-autohide .pip-bubble{animation:pipPop .35s cubic-bezier(.2,1.4,.4,1) both, pipFade .6s ease 9s forwards}
.pip-svg *{transform-box:fill-box;transform-origin:center}
.pip-page,.pip-dots,.pip-sweat,.pip-happy,.pip-spark,.pip-m-o,.pip-m-flat,.pip-brow{opacity:0}
.pip-bob{animation:pipFloat 3.4s ease-in-out infinite}
.pip-eyes{animation:pipBlink 4.6s infinite}
.pip-bulb{animation:pipGlow 2.4s ease-in-out infinite}
.pip-arm-r{transform-origin:0% 50%}
[data-mood=greet] .pip-arm-r{animation:pipWave 1.1s ease-in-out infinite}
[data-mood=reading] .pip-page{opacity:1;animation:pipPage 2.2s ease-in-out infinite}
[data-mood=reading] .pip-pupils{animation:pipScan 1.6s ease-in-out infinite}
[data-mood=reading] .pip-mouth,[data-mood=reading] .pip-cheek{opacity:0}
[data-mood=reading] .pip-bulb,[data-mood=thinking] .pip-bulb{animation-duration:.9s}
[data-mood=thinking] .pip-pupils{transform:translate(-3px,-4px)}
[data-mood=thinking] .pip-dots{opacity:1}
[data-mood=thinking] .pip-dots circle{animation:pipDot 1.2s ease-in-out infinite}
[data-mood=thinking] .pip-dots circle:nth-child(2){animation-delay:.2s}
[data-mood=thinking] .pip-dots circle:nth-child(3){animation-delay:.4s}
[data-mood=talking] .pip-m-smile{opacity:0}
[data-mood=talking] .pip-m-o{opacity:1;animation:pipTalk .32s ease-in-out infinite alternate}
[data-mood=happy] .pip-eyes{opacity:0;animation:none}
[data-mood=happy] .pip-happy,[data-mood=happy] .pip-spark{opacity:1}
[data-mood=happy] .pip-bob{animation:pipBounce .9s cubic-bezier(.3,1.6,.5,1) 2, pipFloat 3.4s ease-in-out 1.8s infinite}
[data-mood=happy] .pip-spark path{animation:pipTwinkle 1.4s ease-in-out infinite}
[data-mood=happy] .pip-spark path:nth-child(2){animation-delay:.5s}
[data-mood=confused] .pip-tilt{transform:rotate(-9deg)}
[data-mood=confused] .pip-sweat,[data-mood=confused] .pip-brow,[data-mood=confused] .pip-m-flat{opacity:1}
[data-mood=confused] .pip-m-smile{opacity:0}
[data-mood=confused] .pip-sweat{animation:pipDrip 2s ease-in infinite}
@keyframes pipFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes pipBounce{0%{transform:translateY(0)}35%{transform:translateY(-16px) scale(1.03,.97)}70%{transform:translateY(0) scale(.98,1.02)}100%{transform:translateY(0)}}
@keyframes pipBlink{0%,92%,100%{transform:scaleY(1)}95%{transform:scaleY(.1)}}
@keyframes pipGlow{0%,100%{opacity:.75}50%{opacity:1}}
@keyframes pipWave{0%,100%{transform:rotate(0)}50%{transform:rotate(-38deg)}}
@keyframes pipScan{0%,100%{transform:translateX(-4px)}50%{transform:translateX(4px)}}
@keyframes pipPage{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-2px) rotate(2deg)}}
@keyframes pipDot{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1)}}
@keyframes pipTalk{from{transform:scaleY(.45)}to{transform:scaleY(1.15)}}
@keyframes pipTwinkle{0%,100%{opacity:.2;transform:scale(.6)}50%{opacity:1;transform:scale(1.1)}}
@keyframes pipDrip{0%{transform:translateY(0);opacity:1}80%{transform:translateY(8px);opacity:.9}100%{transform:translateY(10px);opacity:0}}
@keyframes pipPop{from{opacity:0;transform:translateY(6px) scale(.96)}to{opacity:1;transform:none}}
@keyframes pipFade{to{opacity:0;transform:translateY(4px)}}
@media (max-width:640px){.pip-dock{right:10px;bottom:76px}.pip-dock .pip-svg{width:64px;height:64px}
  .pip-dock .pip-bubble{max-width:190px;font-size:.8rem;margin-bottom:30px}.pip-hero{flex-direction:column;text-align:center}
  .pip-hero .pip-bubble:after{display:none}}
@media (prefers-reduced-motion:reduce){.pip-svg *,.pip-bubble{animation:none!important}}
"""

SVG = (
    '<svg class="pip-svg" viewBox="0 0 140 140" role="img" aria-label="Pip, the Margin mascot">'
    '<defs><linearGradient id="pipBody" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#8B6CFF"/><stop offset="1" stop-color="#38BDF8"/></linearGradient>'
    '<radialGradient id="pipBulb"><stop offset="0" stop-color="#FFE2BD"/><stop offset="1" stop-color="#FF9F5A"/></radialGradient></defs>'
    '<ellipse cx="70" cy="132" rx="30" ry="4.5" fill="#000" opacity=".22"/>'
    '<g class="pip-bob"><g class="pip-tilt">'
    '<g class="pip-dots" fill="#C4B5FD"><circle cx="104" cy="22" r="3.5"/><circle cx="116" cy="14" r="4.5"/><circle cx="130" cy="5" r="5.5"/></g>'
    '<path d="M70 32 Q69 20 77 13" stroke="#B9A8FF" stroke-width="3.2" fill="none" stroke-linecap="round"/>'
    '<circle class="pip-bulb" cx="78" cy="12" r="6.5" fill="url(#pipBulb)"/>'
    '<rect x="15" y="74" width="17" height="11" rx="5.5" fill="#7A63F0"/>'
    '<rect class="pip-arm-r" x="108" y="74" width="17" height="11" rx="5.5" fill="#3FA9EE"/>'
    '<path d="M52 32 H90 L110 52 V96 Q110 118 88 118 H52 Q30 118 30 96 V54 Q30 32 52 32 Z" fill="url(#pipBody)"/>'
    '<path d="M90 32 V46 Q90 52 96 52 H110 Z" fill="#D9D0FF" opacity=".92"/>'
    '<rect x="41" y="56" width="58" height="38" rx="17" fill="#121735"/>'
    '<g class="pip-eyes"><g class="pip-pupils" fill="#E9F6FF">'
    '<ellipse cx="58" cy="75" rx="6" ry="7.5"/><ellipse cx="82" cy="75" rx="6" ry="7.5"/>'
    '<circle cx="60" cy="72" r="1.8" fill="#9AD8FF"/><circle cx="84" cy="72" r="1.8" fill="#9AD8FF"/></g></g>'
    '<g class="pip-happy" stroke="#E9F6FF" stroke-width="3.6" fill="none" stroke-linecap="round">'
    '<path d="M51 78 Q58 68 65 78"/><path d="M75 78 Q82 68 89 78"/></g>'
    '<path class="pip-brow" d="M75 63 L89 60" stroke="#E9F6FF" stroke-width="3" stroke-linecap="round"/>'
    '<g class="pip-cheek" fill="#FFB3C7" opacity=".6"><circle cx="40" cy="103" r="5"/><circle cx="100" cy="103" r="5"/></g>'
    '<g class="pip-mouth" stroke="#1B1F3B" stroke-width="3" fill="none" stroke-linecap="round">'
    '<path class="pip-m-smile" d="M62 104 Q70 111 78 104"/>'
    '<path class="pip-m-flat" d="M62 106 Q66 103 70 106 Q74 109 78 105"/></g>'
    '<ellipse class="pip-m-o" cx="70" cy="106" rx="4.6" ry="4.2" fill="#1B1F3B"/>'
    '<path class="pip-sweat" d="M104 60 Q109 68 104 72 Q99 68 104 60 Z" fill="#7DD3FC"/>'
    '<g class="pip-page"><rect x="44" y="96" width="52" height="34" rx="5" fill="#F5F7FF"/>'
    '<g stroke="#A5B0D6" stroke-width="2.4" stroke-linecap="round"><path d="M52 106 H86"/><path d="M52 113 H80"/><path d="M52 120 H84"/></g></g>'
    '<g class="pip-spark" fill="#FFD38A"><path d="M20 30 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3z"/>'
    '<path d="M118 92 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z"/></g>'
    '</g></g></svg>'
)


def render(mood, message, mode="dock", autohide=False):
    """mode: 'dock' (fixed in the corner) or 'hero' (large, inline on the welcome screen)."""
    mood = mood if mood in MOODS else "idle"
    cls = "pip-dock" + (" pip-autohide" if autohide else "") if mode == "dock" else "pip-hero"
    bubble = f'<div class="pip-bubble"><b class="pip-name">Pip</b>{_html.escape(message)}</div>' if message else ""
    return f'<div class="{cls}" data-mood="{mood}">{SVG}{bubble}</div>'
