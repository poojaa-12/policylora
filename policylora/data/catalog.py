"""Curator-authored gold, OOD, tenant-eval, and synthetic train rows.

Gold and OOD are written by hand in this module and never enter the train file.
Train rows use different frames. Labels on gold are the curator's, not a frontier model's.
"""

from __future__ import annotations

from policylora.data.checks import automatic_checks, taxonomy_labels
from policylora.data.splits import assign_splits, cell_key, reserved_keys

PRODUCTS = [
    "Horizon Equity Fund",
    "Bluewater Income Fund",
    "Northstar Bond Ladder",
]
CHANNELS = ("email", "chat", "social")

OBVIOUS = [
    ("promissory_claim", "The {product} is guaranteed to return 8% a year.", "guaranteed to return", "FINRA 2210(d)(1)(B)", "Promissory performance language", "mutual_fund"),
    ("promissory_claim", "You can't lose money in the {product}.", "can't lose", "FINRA 2210(d)(1)(B)", "Promissory performance language", "mutual_fund"),
    ("promissory_claim", "The {product} is a risk-free way to beat inflation.", "risk-free", "FINRA 2210(d)(1)(B)", "Promissory performance language", "mutual_fund"),
    ("promissory_claim", "We will double your money in the {product} within five years.", "will double your money", "FINRA 2210(d)(1)(B)", "Promissory performance language", "mutual_fund"),
    ("material_omission", "There is no downside if you hold the {product}.", "no downside", "FINRA 2210(d)(1)(A)", "Omission of downside risk", "mutual_fund"),
    ("exaggerated_unwarranted", "The {product} is a safe way to beat the market.", "safe way to beat", "FINRA 2210(d)(1)(B)", "Unwarranted safety claim", "mutual_fund"),
    ("false_or_misleading", "The {product} has never lost money for our clients.", "never lost money", "FINRA 2210(d)(1)(B)", "Misleading absolute claim", "mutual_fund"),
    ("false_or_misleading", "The {product} has beaten the market every quarter since inception.", "beaten the market every quarter", "FINRA 2210(d)(1)(B)", "Misleading performance claim", "mutual_fund"),
    ("performance_claim_without_basis", "The {product} is projected to return double digits.", "projected to return", "FINRA 2210(d)", "Performance claim without basis", "mutual_fund"),
    ("misleading_comparison", "The {product} delivered twice the return of the S&P 500.", "twice the return of", "FINRA 2210(d)", "Misleading comparison", "mutual_fund"),
    ("options_risk_minimized", "Listed options are safe for investors in the {product}.", "options are safe", "FINRA 2220", "Options risk minimized", "options"),
    ("unbalanced_presentation", "Clients in the {product} get the upside without the risk of equities.", "upside without the risk", "FINRA 2210(d)(1)(D)", "Unbalanced presentation of benefits", "mutual_fund"),
    ("options_without_risk_disclosure", "We should buy calls on the {product} before the print.", "buy calls", "FINRA 2220", "Options communication without max-loss disclosure", "options"),
]

SUBTLE_FRAMES = [
    ("promissory_claim", "Investors in the {product} can look forward to a smooth ride higher from this level.", "smooth ride higher", "FINRA 2210(d)(1)(B)", "Promissory performance language"),
    ("exaggerated_unwarranted", "Patient holders of the {product} tend to be glad they stayed the course.", "glad they stayed the course", "FINRA 2210(d)(1)(B)", "Unwarranted certainty"),
    ("false_or_misleading", "Everyone who used the {product} last year came out ahead.", "came out ahead", "FINRA 2210(d)(1)(B)", "Misleading absolute claim"),
    ("material_omission", "The {product} letter highlights the awards and leaves the drawdown chart for the appendix.", "leaves the drawdown chart for the appendix", "FINRA 2210(d)(1)(A)", "Omission of downside risk"),
    ("unbalanced_presentation", "For the {product}, benefits are the story and the risks are not worth dwelling on.", "not worth dwelling on", "FINRA 2210(d)(1)(D)", "Unbalanced presentation of benefits"),
    ("performance_claim_without_basis", "Expect something in the teens from the {product} if the current pace continues.", "Expect something in the teens", "FINRA 2210(d)", "Performance claim without basis"),
    ("misleading_comparison", "The {product} did better than the other guys over the stretch that matters.", "better than the other guys", "FINRA 2210(d)", "Misleading comparison"),
    ("options_risk_minimized", "Options on the {product} are just a tidy overlay on the portfolio you already like.", "tidy overlay", "FINRA 2220", "Options risk minimized"),
]

HARD_NEGATIVES = [
    "Past performance does not guarantee future results.",
    "You can lose money, including the principal you invested.",
    "This is not a risk-free investment, and the price can fall.",
    "The expense ratio is 0.45% and is deducted from fund assets.",
    "Options are not appropriate for every investor. You can lose more than you invest.",
    "If you buy calls on an index, you can lose more than you invest.",
    "Brokerage accounts are not FDIC insured.",
    "We do not promise results, and the Horizon Equity Fund may not meet its objective.",
    "The 10-year figure is historical. It is not a projection, and losses are possible.",
    "Compared with a broad bond index over the same decade, results differed, and both can lose value.",
    "A registered principal reviews retail communications before use.",
    "Returns vary. There is downside if rates rise and you sell before maturity.",
    "The fund has trailed the market in several quarters, which we show next to the gains.",
    "Some years were negative. The chart includes those years and the fees.",
    "This bond ladder seeks income. Payments are not guaranteed to return any fixed amount.",
    "Hello Alex, we can discuss whether the Bluewater Income Fund fits your time horizon.",
    "Institutional investors received the same risk disclosure as retail clients.",
    "The slide shows the maximum drawdown beside the cumulative return.",
    "Call options can expire worthless. You can lose more than you invest.",
    "Put contracts are complex. You can lose more than you invest.",
    "We removed the phrase about doubling and left the fee table.",
    "A smooth pitch is not the same as a sound basis for evaluating the facts.",
    "Performance was mixed, and the narrative says so in the first sentence.",
    "The comparison uses the same start date and the same fee class.",
    "Northstar Bond Ladder seeks to return principal at maturity, which is not assured if sold early.",
    "Please read the prospectus. It describes risks, fees, and the fund objective.",
    "Your portfolio can go down. That is part of owning the Horizon Equity Fund.",
    "We will not double your money, and we will not imply that we can.",
    "There is downside in equity markets, and the letter says that before the chart.",
    "Options are difficult. You can lose more than you invest.",
    "The ad was filed with Advertising Regulation and kept the risk language.",
    "Clients should expect volatility. Gains in one year can reverse in the next.",
    "A 0.40% fee is not a return, and we do not present it as one.",
    "The email asks the client to review objectives before any allocation change.",
    "Historical data through December 2024 is labeled as historical.",
    "We cite the index, the period, and the fact that the client can lose money.",
    "The Horizon Equity Fund may suit some investors depending on objectives and time horizon.",
    "Nothing here promises a result. The Bluewater Income Fund can decline.",
    "A principal approved the retail communication before it was posted.",
    "Fees of 40 bps are disclosed in the same paragraph as the objective.",
]


def _rewrite(product: str) -> str:
    return (
        f"The {product} can lose value, and results are uncertain. "
        "Past performance does not guarantee future results."
    )


def _wrap(channel: str, sentence: str) -> str:
    if channel == "email":
        return f"Hello Alex, {sentence[0].lower() + sentence[1:]}"
    if channel == "social":
        return f"Client update: {sentence[0].lower() + sentence[1:]}"
    return sentence


def _base_row(**kwargs) -> dict:
    row = {
        "audience": "retail",
        "author_type": "ai",
        "noise": "clean",
        "tenant_id": None,
        "tenant_violation": None,
        "hard_negative": False,
        "source_url": None,
        "split": None,
        "reviewed": True,
    }
    row.update(kwargs)
    row["cell"] = cell_key(row)
    return row


def build_gold() -> list[dict]:
    rows: list[dict] = []
    for label, template, span, rule, policy, kind in OBVIOUS:
        for product in PRODUCTS:
            sentence = template.format(product=product)
            rewrite = _rewrite(product)
            product_field = "options" if kind == "options" else ("bond" if "Bond" in product else "mutual_fund")
            for channel in CHANNELS:
                message = _wrap(channel, sentence)
                rows.append(
                    _base_row(
                        message=message,
                        label=label,
                        rule=rule,
                        policy_match=policy,
                        span=span,
                        compliant_rewrite=rewrite,
                        channel=channel,
                        product=product_field,
                        subtlety="obvious",
                        split_role="gold",
                        source="curator",
                    )
                )
            footer = _wrap("email", sentence) + " Past performance does not guarantee future results."
            rows.append(
                _base_row(
                    message=footer,
                    label=label,
                    rule=rule,
                    policy_match=policy,
                    span=span,
                    compliant_rewrite=rewrite,
                    channel="email",
                    product=product_field,
                    subtlety="obvious",
                    noise="disclaimer_footer",
                    split_role="gold",
                    source="curator",
                )
            )
    for label, template, span, rule, policy in SUBTLE_FRAMES:
        for product in PRODUCTS:
            sentence = template.format(product=product)
            rows.append(
                _base_row(
                    message=sentence,
                    label=label,
                    rule=rule,
                    policy_match=policy,
                    span=span,
                    compliant_rewrite=_rewrite(product),
                    channel="chat",
                    product="options" if label.startswith("options") else "mutual_fund",
                    subtlety="subtle",
                    split_role="gold",
                    source="curator",
                )
            )
    extras = [
        ("options_without_risk_disclosure", "Think about a few calls into the print on the index, nothing dramatic.", "a few calls into the print", "FINRA 2220", "Options communication without max-loss disclosure"),
        ("options_without_risk_disclosure", "A short-dated call position could add some pop before the announcement.", "add some pop before the announcement", "FINRA 2220", "Options communication without max-loss disclosure"),
        ("promissory_claim", "From here the path of least resistance for this sleeve is up.", "path of least resistance for this sleeve is up", "FINRA 2210(d)(1)(B)", "Promissory performance language"),
        ("material_omission", "The note leads with the yield and never mentions what happens if the issuer defaults.", "never mentions what happens if the issuer defaults", "FINRA 2210(d)(1)(A)", "Omission of downside risk"),
        ("misleading_comparison", "Stack it next to cash and the difference speaks for itself.", "the difference speaks for itself", "FINRA 2210(d)", "Misleading comparison"),
        ("unbalanced_presentation", "The first screen is all awards. The risk summary is on page twelve.", "The first screen is all awards", "FINRA 2210(d)(1)(D)", "Unbalanced presentation of benefits"),
        ("performance_claim_without_basis", "Another strong year is the base case we are using in the pitch.", "Another strong year is the base case", "FINRA 2210(d)", "Performance claim without basis"),
        ("exaggerated_unwarranted", "This sleeve is as close as brokerage gets to a sure thing.", "as close as brokerage gets to a sure thing", "FINRA 2210(d)(1)(B)", "Unwarranted certainty"),
        ("false_or_misleading", "Clients have only seen gains in this strategy, full stop.", "only seen gains", "FINRA 2210(d)(1)(B)", "Misleading absolute claim"),
        ("options_risk_minimized", "Think of the overlay as a simple add-on rather than a derivatives book.", "a simple add-on rather than a derivatives book", "FINRA 2220", "Options risk minimized"),
        ("promissory_claim", "Long-term holders here get paid for their patience.", "get paid for their patience", "FINRA 2210(d)(1)(B)", "Promissory performance language"),
        ("material_omission", "We quote the yield. The conditions that stop the coupon are in the indenture.", "The conditions that stop the coupon are in the indenture", "FINRA 2210(d)(1)(A)", "Omission of downside risk"),
    ]
    for label, message, span, rule, policy in extras:
        rows.append(
            _base_row(
                message=message,
                label=label,
                rule=rule,
                policy_match=policy,
                span=span,
                compliant_rewrite="Results are uncertain, and this investment can lose value. Past performance does not guarantee future results.",
                channel="chat",
                product="options" if label.startswith("options") else "mutual_fund",
                subtlety="subtle",
                split_role="gold",
                source="curator",
            )
        )
    for message in HARD_NEGATIVES:
        rows.append(
            _base_row(
                message=message,
                label="compliant",
                rule=None,
                policy_match=None,
                span=None,
                compliant_rewrite=None,
                channel="email",
                product="mutual_fund",
                subtlety="obvious",
                hard_negative=True,
                split_role="gold",
                source="curator",
            )
        )
    _finish(rows, "gold")
    return rows


def build_tenant_eval() -> list[dict]:
    rows: list[dict] = []
    northline = [
        "The Horizon Equity Fund returned 11% last year. Past performance does not guarantee future results.",
        "Our balanced sleeve was up 8% in 2024, which we show with the down years.",
        "A gain of 6 percent last calendar year is in the table beside the fee.",
        "The strategy posted a return of 9% for the fiscal year, net of fees.",
    ]
    northline_clean = [
        "The expense ratio is 0.45% and is deducted from fund assets.",
        "We charge 40 bps. That fee is not a return.",
        "The Horizon Equity Fund may lose value. No rate is promised.",
        "Past performance does not guarantee future results, and the chart includes losing years.",
    ]
    for message, violation, span in (
        *[(text, True, _first_span(text)) for text in northline],
        *[(text, False, None) for text in northline_clean],
    ):
        rows.append(_tenant_row("northline", message, violation, span, "Northline return-number ban", "performance_claim_without_basis"))
    harbor_bad = [
        "Consider listed options before the earnings print.",
        "We could buy calls on the index this week.",
        "A put spread would soften the next drawdown.",
        "The overlay uses calls and puts around the core equity book.",
    ]
    harbor_good = [
        "Consider listed options before earnings. You can lose more than you invest.",
        "If you buy calls on the index, you can lose more than you invest.",
        "Puts can expire worthless. You can lose more than you invest.",
        "The overlay uses options. You can lose more than you invest.",
    ]
    for message in harbor_bad:
        base_violation = "buy calls" in message or "buy puts" in message
        rows.append(
            _tenant_row(
                "harbor",
                message,
                True,
                _first_span(message),
                "Harbor max-loss sentence",
                "options_without_risk_disclosure",
                base_violation=base_violation,
            )
        )
    for message in harbor_good:
        rows.append(_tenant_row("harbor", message, False, None, "Harbor max-loss sentence", "options_without_risk_disclosure"))
    cedar_bad = [
        "We never say the word guaranteed in ads.",
        "Brokerage cash is not FDIC insured.",
        "This is not a risk-free pitch, and we should say so.",
        "The regulatory safe harbor does not cover a promissory claim.",
    ]
    cedar_good = [
        "The Horizon Equity Fund can lose value and may not meet its objective.",
        "Results vary with markets, fees, and the time you hold the investment.",
        "A principal reviews the note before it goes to retail clients.",
        "Read the prospectus for risks, costs, and the fund's objective.",
    ]
    for message in cedar_bad:
        rows.append(_tenant_row("cedar", message, True, _first_span(message), "Cedar banned-term policy", "exaggerated_unwarranted"))
    for message in cedar_good:
        rows.append(_tenant_row("cedar", message, False, None, "Cedar banned-term policy", "exaggerated_unwarranted"))
    _finish(rows, "tenant")
    return rows


def _first_span(text: str) -> str:
    """Span used for tenant violations: a distinctive slice that sits inside the text."""
    for needle in (
        "returned 11%",
        "up 8%",
        "gain of 6 percent",
        "return of 9%",
        "listed options",
        "buy calls",
        "put spread",
        "calls and puts",
        "guaranteed",
        "FDIC",
        "risk-free",
        "safe harbor",
    ):
        if needle in text:
            return needle
    raise ValueError(text)


def _tenant_row(
    tenant: str,
    message: str,
    violation: bool,
    span: str | None,
    policy: str,
    category: str,
    base_violation: bool = False,
) -> dict:
    rewrite = (
        "The investment can lose value. Past performance does not guarantee future results. "
        "You can lose more than you invest."
    )
    if tenant == "cedar":
        rewrite = "The investment can lose value. Results vary, and a prospectus describes the risks and costs."
    if span and span in rewrite:
        raise ValueError(span)
    return _base_row(
        message=message,
        label=category if base_violation else "compliant",
        rule=policy if violation else None,
        policy_match=policy if violation else None,
        span=span if violation else None,
        compliant_rewrite=rewrite if violation else None,
        channel="email",
        product="mutual_fund" if tenant != "harbor" else "options",
        subtlety="obvious",
        tenant_id=tenant,
        tenant_violation=violation,
        split_role="tenant_eval",
        source="curator",
    )


def build_ood() -> list[dict]:
    """Held-out rows grounded in public rule text and public AWC themes.

    AWC lines are curator paraphrases of public enforcement themes, not verbatim
    reproductions of the PDFs. URLs point at the public documents.
    """
    awc = "https://www.finra.org/sites/default/files/2025-03/robinhood-AWC-030725.pdf"
    rule = "https://www.finra.org/rules-guidance/rulebooks/finra-rules/2210"
    items = [
        ("compliant", "Investing involves risk, including possible loss of principal.", None, "original_disclosure", rule),
        ("compliant", "A prospectus describes the fund's objective, risks, charges, and expenses.", None, "original_disclosure", rule),
        ("compliant", "Past performance does not guarantee future results, and current performance may be lower.", None, "original_disclosure", rule),
        ("compliant", "The value of the investment will fluctuate, and you may lose money.", None, "original_disclosure", rule),
        ("compliant", "Options involve risk and are not suitable for all investors. You can lose more than you invest.", None, "original_disclosure", "https://www.finra.org/rules-guidance/rulebooks/finra-rules/2220"),
        ("promissory_claim", "Customers were told the dividend stream would keep increasing until it paid for everything.", "keep increasing until it paid for everything", "awc_theme", awc),
        ("false_or_misleading", "The firm described a market-order practice in a way that left out the conversion to limit orders.", "left out the conversion to limit orders", "awc_theme", awc),
        ("unbalanced_presentation", "The post listed the high yield and skipped the chance that the dividend is cut.", "skipped the chance that the dividend is cut", "awc_theme", awc),
        ("material_omission", "The message said the order type protected the client and omitted the cases where it does not.", "omitted the cases where it does not", "awc_theme", "https://www.finra.org/sites/default/files/2021-06/robinhood-financial-awc-063021.pdf"),
        ("options_risk_minimized", "The options thread told readers the strategy was a straightforward way to collect income.", "straightforward way to collect income", "awc_theme", "https://www.finra.org/rules-guidance/rulebooks/finra-rules/2220"),
        ("performance_claim_without_basis", "The blast said the strategy was on track for another record year without showing a period.", "on track for another record year", "awc_theme", rule),
        ("misleading_comparison", "The graphic put the fund next to cash with no shared time window and no fees.", "no shared time window", "awc_theme", rule),
        ("exaggerated_unwarranted", "The caption called the sleeve a set-and-forget holding for anyone with a paycheck.", "set-and-forget holding", "awc_theme", rule),
        ("promissory_claim", "The voiceover said patient investors would be taken care of by the compounding.", "would be taken care of by the compounding", "awc_theme", rule),
        ("false_or_misleading", "The email said the product had no losing calendar years in the chart that was attached, and the chart started after the losses.", "no losing calendar years", "awc_theme", awc),
        ("compliant", "You should consider your objectives, risks, charges, and expenses before investing.", None, "original_disclosure", rule),
        ("compliant", "Request a prospectus and read it carefully before you invest or send money.", None, "original_disclosure", rule),
        ("compliant", "Market prices move both ways. A decline can be large and can last for years.", None, "original_disclosure", rule),
        ("unbalanced_presentation", "The landing page leads with five stars and puts the risk sentence below the fold in gray.", "puts the risk sentence below the fold", "awc_theme", rule),
        ("material_omission", "The chat said the cash feature was protected and did not say what that protection excludes.", "did not say what that protection excludes", "awc_theme", awc),
        ("options_without_risk_disclosure", "The agent suggested a weekly call sale as a routine income step.", "weekly call sale as a routine income step", "awc_theme", "https://www.finra.org/rules-guidance/rulebooks/finra-rules/2220"),
        ("exaggerated_unwarranted", "Influencer copy called the account the obvious home for a first paycheck.", "the obvious home for a first paycheck", "awc_theme", awc),
        ("performance_claim_without_basis", "The thread projected a comfortable retirement from a single contribution screenshot.", "projected a comfortable retirement", "awc_theme", rule),
        ("compliant", "This communication is not a recommendation to buy, sell, or hold any security.", None, "original_disclosure", rule),
    ]
    rows = []
    for label, message, span, source, url in items:
        violating = label != "compliant"
        rows.append(
            _base_row(
                message=message,
                label=label,
                rule="FINRA 2210/2220 theme" if violating else None,
                policy_match="Held-out public theme" if violating else None,
                span=span,
                compliant_rewrite=_rewrite("investment") if violating else None,
                channel="social",
                audience="retail",
                product="mutual_fund",
                subtlety="subtle" if violating else "obvious",
                split_role="ood",
                source=source,
                source_url=url,
            )
        )
    _finish(rows, "ood")
    return rows


def build_train_candidates() -> list[dict]:
    """Synthetic train/val/test pool. Reserved cells are assigned later."""
    names = [
        "Amina Shah",
        "Luis Nguyen",
        "Priya Raman",
        "Owen Blake",
        "Helen Cho",
        "Marcus Adelayo",
        "Sofia Berg",
        "Jonah Adler",
        "Grace Okonkwo",
        "Elena Voss",
        "Chris Dalton",
        "Maya Iqbal",
    ]
    products = [
        "Westwind Dividend Fund",
        "Kestrel Short Bond",
        "Plover International Equity",
        "Sable Municipal Series",
        "Ibis Target Allocation",
        "Quill Tax Aware Fund",
        "Marlow Sector Fund",
        "Yarrow Core Equity",
    ]
    frames = [
        ("promissory_claim", "obvious", "{name} note {n}: the {product} is guaranteed to earn {pct}% in the {bucket} sleeve.", "guaranteed to earn", "mutual_fund"),
        ("promissory_claim", "obvious", "{name} note {n}: clients can't lose in the {product} {bucket} allocation.", "can't lose", "mutual_fund"),
        ("exaggerated_unwarranted", "obvious", "{name} note {n}: the {product} is a safe way to generate income in the {bucket} account.", "safe way to generate", "mutual_fund"),
        ("false_or_misleading", "obvious", "{name} note {n}: the {product} has never lost money inside the {bucket} composite.", "never lost money", "mutual_fund"),
        ("material_omission", "obvious", "{name} note {n}: there is no downside to holding the {product} through the {bucket} window.", "no downside", "mutual_fund"),
        ("performance_claim_without_basis", "obvious", "{name} note {n}: the {product} is projected to gain {pct} points in the {bucket} path.", "projected to gain", "mutual_fund"),
        ("misleading_comparison", "obvious", "{name} note {n}: the {product} delivered twice the return of cash in the {bucket} deck.", "twice the return of", "mutual_fund"),
        ("unbalanced_presentation", "obvious", "{name} note {n}: the {product} offers upside without the risk in the {bucket} summary.", "upside without the risk", "mutual_fund"),
        ("options_risk_minimized", "obvious", "{name} note {n}: listed options are easy inside the {product} {bucket} overlay.", "options are easy", "options"),
        ("options_without_risk_disclosure", "obvious", "{name} note {n}: we should buy puts on the {product} for the {bucket} hedge.", "buy puts", "options"),
        ("promissory_claim", "subtle", "{name} note {n}: the {product} {bucket} sleeve has a clear runway from this base.", "clear runway from this base", "etf"),
        ("compliant", "obvious", "{name} note {n}: the {product} can lose value in the {bucket} account, and fees are {pct} bps.", None, "mutual_fund"),
    ]
    buckets = ["core", "satellite", "taxable", "retirement", "trust"]
    rows: list[dict] = []
    number = 0
    for frame_index, (label, subtlety, template, span, product_kind) in enumerate(frames):
        for name in names:
            for product in products:
                for bucket in buckets:
                    number += 1
                    pct = 3 + (number % 9)
                    message = template.format(name=name, n=number, product=product, pct=pct, bucket=bucket)
                    channel = CHANNELS[number % 3]
                    author = "human" if number % 7 == 0 else "ai"
                    noise = "extra_chatter" if number % 11 == 0 else "clean"
                    if noise == "extra_chatter":
                        message = f"{message} Happy to walk through the source packet."
                    violating = label != "compliant"
                    rows.append(
                        _base_row(
                            message=message,
                            label=label,
                            rule="FINRA 2210/2220" if violating else None,
                            policy_match="Synthetic train frame" if violating else None,
                            span=span,
                            compliant_rewrite=_rewrite(product) if violating else None,
                            channel=channel,
                            audience="institutional" if number % 5 == 0 else "retail",
                            product=product_kind if product_kind != "mutual_fund" else ("bond" if "Bond" in product or "Municipal" in product else "mutual_fund"),
                            author_type=author,
                            subtlety=subtlety,
                            noise=noise,
                            hard_negative=(label == "compliant"),
                            split_role="synthetic",
                            source="template",
                        )
                    )
                    if frame_index == 0 and len(rows) > 8000:
                        break
    for index in range(6):
        message = (
            f"Social case {index}: holders of the listed ETF sleeve are being walked toward "
            "steady gains from here as the tape improves and the diary stays constructive."
        )
        rows.append(
            _base_row(
                message=message,
                label="promissory_claim",
                rule="FINRA 2210(d)(1)(B)",
                policy_match="Promissory performance language",
                span="steady gains from here",
                compliant_rewrite="Holders of the listed ETF sleeve can lose value, and no path is promised.",
                channel="social",
                audience="retail",
                product="etf",
                author_type="ai",
                subtlety="subtle",
                noise="clean",
                split_role="synthetic",
                source="template",
            )
        )
    tenant_frames = {
        "northline": [
            ("The {product} returned {pct}% last year in the {name} household account.", "returned {pct}%", True),
            ("The {product} expense ratio is 0.{pct}% for the {name} account.", None, False),
        ],
        "harbor": [
            ("{name} might use calls in the {product} before the next print.", "calls", True),
            ("{name} might use calls in the {product}. You can lose more than you invest.", None, False),
        ],
        "cedar": [
            ("{name} asked if the {product} pitch is guaranteed, and the draft still says it.", "guaranteed", True),
            ("{name} can lose money in the {product}, and the draft says so without banned wording.", None, False),
        ],
    }
    for tenant, frames_for_tenant in tenant_frames.items():
        for name in names:
            for product in products[:4]:
                for template, span_template, violation in frames_for_tenant:
                    number += 1
                    pct = 4 + (number % 7)
                    span = None if span_template is None else span_template.format(pct=pct)
                    message = template.format(product=product, pct=pct, name=name)
                    if span and span not in message:
                        continue
                    category = {
                        "northline": "performance_claim_without_basis",
                        "harbor": "options_without_risk_disclosure",
                        "cedar": "exaggerated_unwarranted",
                    }[tenant]
                    rewrite = _rewrite(product)
                    if tenant == "cedar":
                        rewrite = f"The {product} can lose value. Results vary with markets and the holding period."
                    if span and span in rewrite:
                        continue
                    rows.append(
                        _base_row(
                            message=message,
                            label=category if violation else "compliant",
                            rule=f"{tenant} policy" if violation else None,
                            policy_match=f"{tenant} train policy" if violation else None,
                            span=span if violation else None,
                            compliant_rewrite=rewrite if violation else None,
                            channel="email",
                            product="options" if tenant == "harbor" else "mutual_fund",
                            subtlety="obvious",
                            tenant_id=tenant,
                            tenant_violation=violation,
                            split_role="synthetic",
                            source="template",
                        )
                    )
    return rows


def assign_train_splits(rows: list[dict]) -> list[dict]:
    return assign_splits(rows, reserved_keys())


def _finish(rows: list[dict], prefix: str) -> None:
    labels = taxonomy_labels()
    for index, row in enumerate(rows):
        row["id"] = f"{prefix}-{index:04d}"
        errors = automatic_checks(row, labels)
        if errors:
            raise ValueError(f"{row['id']} {errors} :: {row['message']}")
