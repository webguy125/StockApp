"""
Curated Mid-Cap and Small-Cap Stocks with Active Options Markets
Organized by sector for balanced training set

Selection Criteria:
- S&P MidCap 400 or SmallCap 600 constituents (or equivalent quality)
- Active options market (high open interest)
- Average daily volume > 500K shares
- Liquid options (tight bid-ask spreads)
- Clean historical data
- Strong institutional following
"""

# Format: 'SYMBOL': 'expected_tier' (M=mid, S=small)
# We'll verify actual market caps with yfinance

CURATED_STOCKS = {
    # ================================================================
    # TECHNOLOGY (need: 7 mid, 7 small)
    # ================================================================
    'technology': {
        'mid_cap': [
            'SNOW',   # Snowflake - Cloud Data
            'PLTR',   # Palantir - AI/Data Analytics
            'DDOG',   # Datadog - Monitoring
            'OKTA',   # Okta - Identity Management
            'ZS',     # Zscaler - Cybersecurity
            'CRWD',   # CrowdStrike - Cybersecurity (may be large now)
            'NET',    # Cloudflare - Edge Computing
            'TWLO',   # Twilio - Communications API
            'MDB',    # MongoDB - Database
            'HUBS',   # HubSpot - Marketing Software
        ],
        'small_cap': [
            'FROG',   # JFrog - DevOps
            'BILL',   # Bill.com - Payments
            'S',      # SentinelOne - Cybersecurity
            'GTLB',   # GitLab - DevOps
            'DT',     # Dynatrace - Monitoring
            'ESTC',   # Elastic - Search/Analytics
            'SMCI',   # Super Micro Computer (may be mid now)
            'CVLT',   # Commvault - Data Management
        ]
    },

    # ================================================================
    # COMMUNICATION SERVICES (need: 7 mid, 7 small)
    # ================================================================
    'communication_services': {
        'mid_cap': [
            'PINS',   # Pinterest - Social Media
            'SNAP',   # Snap Inc - Social Media
            'ROKU',   # Roku - Streaming Platform
            'MTCH',   # Match Group - Dating Apps
            'IAC',    # IAC - Internet Holding Company
            'PARA',   # Paramount Global - Media
            'WBD',    # Warner Bros Discovery - Media
        ],
        'small_cap': [
            'BMBL',   # Bumble - Dating App
            'FUBO',   # FuboTV - Streaming
            'SIRI',   # SiriusXM - Satellite Radio
            'NWSA',   # News Corp - Media
            'NYT',    # New York Times - Media
            'MSGN',   # MSG Networks - Sports Media
            'DISCA',  # Discovery (if still separate)
        ]
    },

    # ================================================================
    # CONSUMER DISCRETIONARY (need: 2 mid, 6 small)
    # ================================================================
    'consumer_discretionary': {
        'mid_cap': [
            'ETSY',   # Etsy - E-commerce
            'W',      # Wayfair - Home Goods E-commerce
            'BKNG',   # Booking Holdings (may be large)
            'RCL',    # Royal Caribbean - Cruise
            'LVS',    # Las Vegas Sands - Gaming
        ],
        'small_cap': [
            'CHWY',   # Chewy - Pet E-commerce
            'PRTY',   # Party City - Retail
            'BBWI',   # Bath & Body Works - Retail
            'URBN',   # Urban Outfitters - Retail
            'ANF',    # Abercrombie & Fitch - Apparel
            'CPRI',   # Capri Holdings - Luxury
            'RL',     # Ralph Lauren - Apparel (may be mid)
            'TPR',    # Tapestry - Luxury
        ]
    },

    # ================================================================
    # CONSUMER STAPLES (need: 7 mid, 7 small)
    # ================================================================
    'consumer_staples': {
        'mid_cap': [
            'BG',     # Bunge - Agriculture
            'TSN',    # Tyson Foods - Food Producer
            'CAG',    # Conagra - Food Producer
            'SJM',    # Smucker - Food Producer
            'K',      # Kellogg - Food Producer
            'GIS',    # General Mills - Food Producer
            'HSY',    # Hershey - Confectionery
            'CPB',    # Campbell Soup - Food Producer
        ],
        'small_cap': [
            'HRL',    # Hormel Foods - Food Producer
            'MKC',    # McCormick - Spices
            'LW',     # Lamb Weston - Frozen Foods
            'POST',   # Post Holdings - Cereals
            'INGR',   # Ingredion - Food Ingredients
            'FLO',    # Flowers Foods - Bakery
            'CAL',    # Caleres - Footwear
            'BGS',    # B&G Foods - Packaged Foods
        ]
    },

    # ================================================================
    # FINANCIALS (need: 6 mid, 7 small)
    # ================================================================
    'financials': {
        'mid_cap': [
            'SOFI',   # SoFi - Fintech
            'AFRM',   # Affirm - Buy Now Pay Later
            'LC',     # LendingClub - Fintech
            'ALLY',   # Ally Financial - Auto Lending
            'SYF',    # Synchrony Financial - Consumer Finance
            'DFS',    # Discover Financial - Credit Cards
            'KEY',    # KeyCorp - Regional Bank
            'RF',     # Regions Financial - Regional Bank
            'HBAN',   # Huntington Bancshares - Regional Bank
        ],
        'small_cap': [
            'EWBC',   # East West Bancorp - Regional Bank
            'WAL',    # Western Alliance - Regional Bank
            'PACW',   # PacWest Bancorp - Regional Bank
            'FHN',    # First Horizon - Regional Bank
            'ONB',    # Old National Bank - Regional Bank
            'UMBF',   # UMB Financial - Regional Bank
            'SBNY',   # Signature Bank (verify not delisted)
            'OZK',    # Bank OZK - Regional Bank
        ]
    },

    # ================================================================
    # HEALTHCARE (need: 5 mid, 7 small)
    # ================================================================
    'healthcare': {
        'mid_cap': [
            'DXCM',   # DexCom - Medical Devices
            'ALGN',   # Align Technology - Medical Devices (may be large)
            'PODD',   # Insulet - Medical Devices
            'TDOC',   # Teladoc - Telehealth
            'VEEV',   # Veeva Systems - Healthcare Software
            'EXAS',   # Exact Sciences - Diagnostics
            'TECH',   # Bio-Techne - Life Sciences
            'INCY',   # Incyte - Biotech
        ],
        'small_cap': [
            'NTRA',   # Natera - Diagnostics
            'IONS',   # Ionis Pharmaceuticals - Biotech
            'UTHR',   # United Therapeutics - Biotech
            'NBIX',   # Neurocrine Bio - Biotech
            'ALNY',   # Alnylam Pharma - Biotech
            'BMRN',   # BioMarin - Biotech
            'SRPT',   # Sarepta - Biotech
            'RARE',   # Ultragenyx - Biotech
        ]
    },

    # ================================================================
    # INDUSTRIALS (need: 5 mid, 7 small)
    # ================================================================
    'industrials': {
        'mid_cap': [
            'CARR',   # Carrier Global - HVAC
            'OTIS',   # Otis Worldwide - Elevators (may be large)
            'URI',    # United Rentals - Equipment Rental
            'JBHT',   # JB Hunt - Trucking
            'CHRW',   # CH Robinson - Logistics
            'EXPD',   # Expeditors - Logistics
            'ODFL',   # Old Dominion - Trucking
        ],
        'small_cap': [
            'R',      # Ryder System - Transportation
            'JBLU',   # JetBlue - Airline
            'ALK',    # Alaska Air - Airline
            'SAVE',   # Spirit Airlines - Airline
            'MATX',   # Matson - Shipping
            'HUBG',   # Hub Group - Logistics
            'SAIA',   # Saia - Trucking
            'ARCB',   # ArcBest - Logistics
        ]
    },

    # ================================================================
    # ENERGY (need: 7 mid, 7 small)
    # ================================================================
    'energy': {
        'mid_cap': [
            'FANG',   # Diamondback Energy - Oil & Gas
            'DVN',    # Devon Energy - Oil & Gas
            'OXY',    # Occidental Petroleum - Oil & Gas
            'MRO',    # Marathon Oil - Oil & Gas
            'HES',    # Hess - Oil & Gas
            'EQT',    # EQT Corp - Natural Gas
            'CTRA',   # Coterra Energy - Oil & Gas
            'VLO',    # Valero - Refining
        ],
        'small_cap': [
            'SM',     # SM Energy - Oil & Gas
            'MGY',    # Magnolia Oil & Gas
            'MTDR',   # Matador Resources - Oil & Gas
            'PR',     # Permian Resources
            'CHRD',   # Chord Energy
            'AR',     # Antero Resources - Natural Gas
            'RRC',    # Range Resources - Natural Gas
            'CNX',    # CNX Resources - Natural Gas
        ]
    },

    # ================================================================
    # MATERIALS (need: 3 mid, 6 small)
    # ================================================================
    'materials': {
        'mid_cap': [
            'FCX',    # Freeport-McMoRan - Copper (may be large)
            'ALB',    # Albemarle - Lithium
            'MP',     # MP Materials - Rare Earths
            'STLD',   # Steel Dynamics - Steel
            'CF',     # CF Industries - Fertilizer
        ],
        'small_cap': [
            'X',      # US Steel - Steel
            'CLF',    # Cleveland-Cliffs - Steel
            'NUE',    # Nucor - Steel (may be mid)
            'CMC',    # Commercial Metals - Steel
            'RS',     # Reliance Steel - Steel Distribution
            'HCC',    # Warrior Met Coal
            'CENX',   # Century Aluminum
            'TECK',   # Teck Resources - Mining
        ]
    },

    # ================================================================
    # UTILITIES (need: 6 mid, 7 small)
    # ================================================================
    'utilities': {
        'mid_cap': [
            'ES',     # Eversource Energy - Electric
            'CMS',    # CMS Energy - Utility
            'AEE',    # Ameren - Utility
            'LNT',    # Alliant Energy - Utility
            'EVRG',   # Evergy - Utility
            'NI',     # NiSource - Gas Utility
            'PNW',    # Pinnacle West - Electric
        ],
        'small_cap': [
            'OGE',    # OGE Energy - Utility
            'NWE',    # Northwestern Energy - Utility
            'AVA',    # Avista - Utility
            'POR',    # Portland General - Electric
            'NWN',    # Northwest Natural - Gas Utility
            'SJW',    # SJW Group - Water Utility
            'AWR',    # American States Water - Water
            'CWT',    # California Water - Water Utility
        ]
    },

    # ================================================================
    # REAL ESTATE (need: 7 mid, 7 small)
    # ================================================================
    'real_estate': {
        'mid_cap': [
            'INVH',   # Invitation Homes - Residential REIT
            'EXR',    # Extra Space Storage - Storage REIT
            'AVB',    # AvalonBay - Residential REIT
            'EQR',    # Equity Residential - REIT
            'VTR',    # Ventas - Healthcare REIT
            'ARE',    # Alexandria Real Estate - Life Science REIT
            'DLR',    # Digital Realty - Data Center REIT
            'PSA',    # Public Storage - Storage REIT
        ],
        'small_cap': [
            'CUBE',   # CubeSmart - Storage REIT
            'LSI',    # Life Storage - Storage REIT
            'NSA',    # National Storage - Storage REIT
            'REXR',   # Rexford Industrial - Industrial REIT
            'FR',     # First Industrial - Industrial REIT
            'TRNO',   # Terreno Realty - Industrial REIT
            'EGP',    # EastGroup Properties - Industrial REIT
            'STAG',   # STAG Industrial - Industrial REIT
        ]
    }
}


def print_summary():
    """Print summary of curated stocks"""
    print("=" * 80)
    print("CURATED MID-CAP AND SMALL-CAP STOCKS BY SECTOR")
    print("=" * 80)
    print()

    total_mid = 0
    total_small = 0

    for sector in sorted(CURATED_STOCKS.keys()):
        mid_count = len(CURATED_STOCKS[sector]['mid_cap'])
        small_count = len(CURATED_STOCKS[sector]['small_cap'])
        total_mid += mid_count
        total_small += small_count

        print(f"{sector:30s} Mid: {mid_count:2d}, Small: {small_count:2d}")

    print()
    print(f"Total Mid-Cap Candidates:   {total_mid}")
    print(f"Total Small-Cap Candidates: {total_small}")
    print(f"Total Candidates:           {total_mid + total_small}")
    print()
    print("Next step: Verify market caps and options availability with yfinance")


if __name__ == '__main__':
    print_summary()
