EXTRACTION_PROMPT = """You are a financial analyst tasked with extracting key data from financial statements.
Please extract the following information from the provided document and output it in JSON format:

```json
{
"company_name": "",
"reporting_period": "",
"currency": "",
"industry": "",
"income_statement": {
  "net_sales": null,
  "cost_of_goods_sold": null,
  "gross_profit": null,
  "operating_expenses": null,
  "operating_income": null,
  "interest_expenses": null,
  "net_income": null,
  "previous_year_sales": null,
  "previous_year_net_income": null
},
"balance_sheet": {
  "cash_and_equivalents": null,
  "current_assets": null,
  "total_assets": null,
  "current_liabilities": null,
  "total_liabilities": null,
  "shareholders_equity": null,
  "average_inventory": null,
  "average_accounts_receivable": null,
  "previous_year_total_assets": null,
  "previous_year_total_liabilities": null
},
"cash_flow": {
  "operating_cash_flow": null,
  "capital_expenditures": null,
  "free_cash_flow": null
},
"notes": {
  "adj_ebitda_available": false,
  "adj_ebitda_details": "",
  "adj_working_capital_available": false,
  "adj_working_capital_details": "",
  "risk_factors": [],
  "significant_events": []
}
}
```

If any information is not available, use null for that value.
If numbers have units (like thousands or millions), make sure to convert them to actual numbers and not include the units in the JSON values.
If you see values for multiple years, extract both the current year and previous year data where indicated.
For average values (like average inventory), calculate them if provided with beginning and ending values, or use the most recent value if only one is available.
For risk factors and significant events, extract any mentions of major risks, unusual transactions, legal issues, or significant business events.
If you can identify the industry the company operates in, include it in the "industry" field.

Remember to format your response ONLY as valid JSON within the ```json and ``` tags. Do not add any additional explanation before or after the JSON."""

OVERVIEW_PROMPT = """Based on the extracted financial data, provide a detailed business overview. 
Include the following sections clearly:

1. COMPANY PROFILE:
   - Core business activities and main products/services
   - Industry and market position
   - Size and scale of operations

2. LEADERSHIP & GOVERNANCE:
   - CEO and key executive names if mentioned
   - Board composition if available
   - Ownership structure (public/private, major shareholders)

3. RECENT DEVELOPMENTS:
   - Major business events or changes in the reporting period
   - Acquisitions, restructuring, or strategic shifts
   - Significant changes in financial figures with potential reasons (e.g., revenue increases/decreases)

4. FINANCIAL HIGHLIGHTS:
   - Brief summary of the most important financial metrics
   - Notable trends in the data

The data provided is:
{extracted_data}

Format your response as continuous paragraphs with appropriate section headings. If specific information is unavailable, briefly acknowledge this rather than making assumptions."""

FINDINGS_PROMPT = """As a financial analyst, provide detailed key findings and insights based on the following financial data:

Extracted Data:
{extracted_data}

Calculated Ratios:
{calculated_ratios}

Please include these specific sections:

1. EXECUTIVE SUMMARY: 2-3 paragraphs highlighting the most critical insights

2. PROFITABILITY ANALYSIS:
   - Detailed assessment of gross margin, operating margin, ROA, and ROE
   - Whether each metric is strong or weak compared to typical standards
   - Likely causes for the observed performance
   - Recommendations for improvement

3. LIQUIDITY & SOLVENCY ASSESSMENT:
   - Analysis of current ratio, cash ratio, debt ratio, and interest coverage
   - Risk evaluation based on these metrics
   - Recommendations for optimal capital structure

4. EFFICIENCY EVALUATION:
   - Analysis of asset turnover, inventory turnover, and receivables turnover
   - Operational improvements needed
   - Industry comparisons where possible

5. NOTABLE TRENDS:
   - Year-over-year changes (if data available)
   - Unusual patterns or anomalies
   - Correlation between different financial metrics

For each section, explain the real-world business implications of the numbers, potential causes, and actionable insights.

Return your complete response as valid JSON with the following structure:
{
  "key_findings": "<Your formatted analysis as a string with proper formatting>",
  "red_flags": [<List of red flag objects with "issue", "severity", and "recommendation" fields>]
}"""

SENTIMENT_PROMPT = """Analyze the management commentary and tone in the provided financial document and provide a detailed sentiment analysis.

Focus on:
1. OVERALL TONE: Is management primarily optimistic, neutral, cautious, or negative?
2. KEY PHRASES: What specific language indicates their outlook?
3. FORWARD-LOOKING STATEMENTS: How do they characterize future prospects?
4. RISK DISCLOSURE: How transparent are they about challenges?
5. CONSISTENCY: Does their tone match the actual financial results?

The financial data is:
{extracted_data}

Format your response as a structured analysis with clear sections. Use specific examples from the text when available. If management commentary is limited, note this limitation and base your assessment on the available information. Provide a balanced assessment that would be valuable to investors."""

BUSINESS_MODEL_PROMPT = """Based on this financial data, recommend 2-3 innovative business model enhancements or pivots that could benefit the company.

Financial Data:
{extracted_data}

For each recommendation:
1. CONCEPT: Describe a specific business model innovation aligned with the company's industry and financial position.
2. STRATEGIC RATIONALE: Explain why this model makes sense given their financial strengths/weaknesses.
3. IMPLEMENTATION APPROACH: Outline key steps for adoption.
4. EXPECTED FINANCIAL IMPACT: Project potential effects on revenue, margins, and overall financial health.
5. RISK CONSIDERATIONS: Identify implementation challenges and mitigation strategies.

Focus on actionable, realistic recommendations based on observable financial patterns. If the company shows strong liquidity but poor margins, consider models to enhance profitability. If growth is stalling, suggest expansion strategies.

Format your response with clear headings and bullet points for each recommendation."""
