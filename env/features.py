import re
import urllib.parse


class URLReputation:
    SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "rebrand.ly"}
    FREE_TLDS = {".tk", ".ml", ".cf", ".ga", ".gq"}
    BRAND_NAMES = ["paypal", "apple", "amazon", "microsoft", "google", "facebook", "netflix", "instagram"]

    def analyze(self, url: str) -> dict:
        flags = []
        score = 0.0
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full = url.lower()

        if parsed.scheme != "https":
            flags.append("no_https")
            score += 0.2

        if any(domain == s or domain.endswith("." + s) for s in self.SHORTENERS):
            flags.append("url_shortener")
            score += 0.3

        if any(full.endswith(tld) or (tld + "/") in full for tld in self.FREE_TLDS):
            flags.append("free_tld")
            score += 0.4

        if re.search(r"\d{1,3}(\.\d{1,3}){3}", domain):
            flags.append("ip_address")
            score += 0.5

        if "@" in url or "%40" in url:
            flags.append("at_sign")
            score += 0.5

        subdomain_parts = domain.split(".")
        if len(subdomain_parts) > 3:
            flags.append("excessive_subdomains")
            score += 0.2

        for brand in self.BRAND_NAMES:
            if brand in domain and not domain.endswith(f"{brand}.com"):
                flags.append(f"brand_impersonation:{brand}")
                score += 0.5
                break

        if len(url) > 100:
            flags.append("long_url")
            score += 0.1

        if re.search(r"(login|verify|secure|update|confirm|account|banking)", path):
            flags.append("suspicious_path_keywords")
            score += 0.2

        score = round(min(1.0, score), 4)
        is_suspicious = score > 0.4
        confidence = round(score if is_suspicious else 1.0 - score, 4)
        return {
            "url": url,
            "domain": domain,
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "risk_score": score,
            "flags": flags,
            "label": "suspicious" if is_suspicious else "clean",
        }


class EmailHeaderAnalyzer:
    def analyze(self, headers: str) -> dict:
        flags = []
        score = 0.0
        lower = headers.lower()

        if "spf=fail" in lower or "spf=softfail" in lower:
            flags.append("spf_fail")
            score += 0.4

        if "dkim=fail" in lower:
            flags.append("dkim_fail")
            score += 0.4

        if "dmarc=fail" in lower:
            flags.append("dmarc_fail")
            score += 0.4

        reply_to = re.search(r"reply-to:\s*(.+)", lower)
        from_match = re.search(r"from:\s*(.+)", lower)
        if reply_to and from_match:
            if reply_to.group(1).strip() != from_match.group(1).strip():
                flags.append("reply_to_mismatch")
                score += 0.3

        if re.search(r"x-mailer:\s*(bulk|mass|blast)", lower):
            flags.append("bulk_mailer")
            score += 0.2

        if re.search(r"received:.*\d{1,3}(\.\d{1,3}){3}.*\d{1,3}(\.\d{1,3}){3}", lower):
            flags.append("multiple_relay_hops")
            score += 0.1

        score = round(min(1.0, score), 4)
        is_suspicious = score > 0.4
        confidence = round(score if is_suspicious else 1.0 - score, 4)
        return {
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "risk_score": score,
            "flags": flags,
            "label": "suspicious" if is_suspicious else "clean",
        }
