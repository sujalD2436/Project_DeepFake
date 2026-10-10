from __future__ import annotations

from typing import Any


def render_verification_report_html(session: dict[str, Any], is_valid: bool, message: str) -> str:
    """Render a mobile-friendly reviewer verification report HTML page."""
    session_id = session.get("session_id", "Unknown")
    status = session.get("status", "PENDING").upper()
    clip_score = session.get("clip_score")
    usable_frames = session.get("usable_frames", 0)
    created_at = session.get("created_at", "N/A")
    verified_at = session.get("verified_at", "Pending Review")
    expires_at = session.get("expires_at", "N/A")
    explanation_base64 = session.get("explanation_png_base64")
    reviewer_verdict = session.get("reviewer_verdict", "Unassigned")
    is_expired = session.get("is_expired", False)

    # Status color scheme
    if is_expired:
        badge_class = "badge-expired"
        badge_text = "EXPIRED TOKEN"
    elif not is_valid:
        badge_class = "badge-danger"
        badge_text = "INVALID"
    elif status == "LOWER_SCORE_DEMO_ONLY":
        badge_class = "badge-success"
        badge_text = "LOW RISK DEMO SIGNAL"
    elif status == "MANUAL_REVIEW":
        badge_class = "badge-danger"
        badge_text = "FLAGGED FOR REVIEW"
    else:
        badge_class = "badge-warning"
        badge_text = status.replace("_", " ")

    score_display = f"{clip_score:.1%}" if clip_score is not None else "N/A"
    heatmap_html = ""
    if explanation_base64:
        heatmap_html = f"""
        <div class="card section">
            <h3>Model Attribution (Grad-CAM)</h3>
            <p class="caption">Regions influencing classifier score (not a manipulation mask)</p>
            <div class="heatmap-container">
                <img src="data:image/png;base64,{explanation_base64}" alt="Grad-CAM Heatmap" />
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KYC Reviewer Verification Report - {session_id}</title>
    <style>
        :root {{
            --bg: #0B0F19;
            --card-bg: #111827;
            --card-border: #1F2937;
            --text-main: #F9FAFB;
            --text-muted: #9CA3AF;
            --accent: #3B82F6;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 20px 16px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 580px;
            width: 100%;
        }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        .header p {{
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-top: 4px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }}
        .badge {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #10B981; }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid #F59E0B; }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444; }}
        .badge-expired {{ background: rgba(156, 163, 175, 0.15); color: #9CA3AF; border: 1px solid #6B7280; }}
        .meta-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.92rem;
        }}
        .meta-row:last-child {{ border-bottom: none; }}
        .meta-label {{ color: var(--text-muted); }}
        .meta-value {{ font-weight: 600; font-family: monospace; }}
        .score-box {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 16px;
            margin: 14px 0 6px 0;
        }}
        .score-val {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--accent);
        }}
        .section h3 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .caption {{
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-bottom: 12px;
        }}
        .heatmap-container {{
            text-align: center;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
        }}
        .heatmap-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .security-banner {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.82rem;
            color: #93C5FD;
            margin-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Fusion KYC Reviewer Report</h1>
            <p>Session Authenticity & Deepfake Risk Verification</p>
        </div>

        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <span style="font-size: 0.85rem; color: var(--text-muted);">Status Verdict</span>
                <span class="badge {badge_class}">{badge_text}</span>
            </div>

            <div class="score-box">
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Fake-Like Model Score</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">{usable_frames} usable face frames evaluated</div>
                </div>
                <div class="score-val">{score_display}</div>
            </div>

            <div class="meta-row">
                <span class="meta-label">Session ID</span>
                <span class="meta-value">{session_id}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Created At</span>
                <span class="meta-value">{created_at}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Verified At</span>
                <span class="meta-value">{verified_at}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Expires At</span>
                <span class="meta-value">{expires_at}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Audit Reviewer Verdict</span>
                <span class="meta-value">{reviewer_verdict}</span>
            </div>
        </div>

        {heatmap_html}

        <div class="security-banner">
            <span>🔒</span>
            <div>
                <strong>Security Guarantee:</strong> This QR verification token is ephemeral and expiring. Zero private identity information is encoded inside the QR matrix.
            </div>
        </div>
    </div>
</body>
</html>"""
