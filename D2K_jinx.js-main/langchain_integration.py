from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json
import re
import logging
import time
from financial_tools import (
    calculate_current_ratio, calculate_debt_to_equity_ratio,
    calculate_gross_margin_ratio, calculate_operating_margin_ratio,
    calculate_return_on_assets_ratio, calculate_return_on_equity_ratio,
    calculate_asset_turnover_ratio, calculate_inventory_turnover_ratio,
    calculate_receivables_turnover_ratio, calculate_debt_ratio,
    calculate_interest_coverage_ratio
)
from prompts import EXTRACTION_PROMPT, OVERVIEW_PROMPT, FINDINGS_PROMPT

load_dotenv()

class LangChainHandler:
    """Handles multi-step LLM operations for financial analysis."""
    
    def __init__(self):
        # Use gemini-2.0-flash for data extraction and gemini-2.0-flash-thinking for reasoning
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        self.extraction_llm = GoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            google_api_key=api_key
        )
        self.analysis_llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21", 
            temperature=0.2,
            google_api_key=api_key
        )
        
        # Setup the extraction chain
        self._setup_extraction_chain()
        
        # Setup the analysis chains
        self._setup_analysis_chains()
        
        # NEW: Setup extra chains for sentiment and business model
        from prompts import SENTIMENT_PROMPT, BUSINESS_MODEL_PROMPT
        self.sentiment_prompt = ChatPromptTemplate.from_template(SENTIMENT_PROMPT)
        self.sentiment_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.sentiment_prompt,
            output_key="sentiment_analysis",
            verbose=True
        )
        self.business_model_prompt = ChatPromptTemplate.from_template(BUSINESS_MODEL_PROMPT)
        self.business_model_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.business_model_prompt,
            output_key="business_model",
            verbose=True
        )
        
    def _setup_extraction_chain(self):
        """Set up the extraction chain for financial data"""
        extraction_template = EXTRACTION_PROMPT
        self.extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
        self.extraction_chain = LLMChain(
            llm=self.extraction_llm, 
            prompt=self.extraction_prompt, 
            output_key="extracted_data",
            verbose=True
        )
    
    def _setup_analysis_chains(self):
        """Set up chains for business overview and key findings"""
        overview_template = OVERVIEW_PROMPT
        self.overview_prompt = ChatPromptTemplate.from_template(overview_template)
        self.overview_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.overview_prompt,
            output_key="business_overview",
            verbose=True
        )
        
        findings_template = FINDINGS_PROMPT
        self.findings_prompt = ChatPromptTemplate.from_template(findings_template)
        self.findings_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.findings_prompt,
            output_key="key_findings",
            verbose=True
        )
    
    def calculate_financial_ratios(self, data: Dict) -> Dict:
        """Calculate financial ratios using the extracted data"""
        # Extract relevant financial data
        income_statement = data.get('income_statement', {})
        balance_sheet = data.get('balance_sheet', {})
        cash_flow = data.get('cash_flow', {})
        
        # Values needed for ratio calculations
        net_sales = income_statement.get('net_sales', 0)
        cost_of_goods_sold = income_statement.get('cost_of_goods_sold', 0)
        gross_profit = income_statement.get('gross_profit', 0)
        operating_income = income_statement.get('operating_income', 0)
        interest_expenses = income_statement.get('interest_expenses', 0)
        net_income = income_statement.get('net_income', 0)
        previous_year_sales = income_statement.get('previous_year_sales', 0)
        previous_year_net_income = income_statement.get('previous_year_net_income', 0)
        
        cash_and_equivalents = balance_sheet.get('cash_and_equivalents', 0)
        current_assets = balance_sheet.get('current_assets', 0)
        total_assets = balance_sheet.get('total_assets', 0)
        current_liabilities = balance_sheet.get('current_liabilities', 0)
        total_liabilities = balance_sheet.get('total_liabilities', 0)
        shareholders_equity = balance_sheet.get('shareholders_equity', 0)
        average_inventory = balance_sheet.get('average_inventory', 0)
        average_accounts_receivable = balance_sheet.get('average_accounts_receivable', 0)
        previous_year_total_assets = balance_sheet.get('previous_year_total_assets', 0)
        previous_year_total_liabilities = balance_sheet.get('previous_year_total_liabilities', 0)
        
        operating_cash_flow = cash_flow.get('operating_cash_flow', 0)
        capital_expenditures = cash_flow.get('capital_expenditures', 0)
        free_cash_flow = cash_flow.get('free_cash_flow', 0)
        
        # Use net_sales for net_credit_sales if not provided
        net_credit_sales = net_sales
        
        # Calculate average_total_assets as total_assets if not provided
        average_total_assets = total_assets
        
        # Helper function to safely calculate ratios
        def safe_calculate(func, numerator, denominator, default=None):
            try:
                if denominator == 0:
                    return default
                return func(numerator, denominator)
            except Exception:
                return default
        
        # Standard ratios calculation (existing code)
        ratios = {
            "Current Ratio": {
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "ratio_value": safe_calculate(calculate_current_ratio, current_assets, current_liabilities, "N/A")
            },
            "Cash Ratio": {
                "cash_and_equivalents": cash_and_equivalents,
                "current_liabilities": current_liabilities,
                "ratio_value": safe_calculate(
                    lambda cash, liab: cash / liab if liab != 0 else "N/A",
                    cash_and_equivalents, current_liabilities, "N/A"
                )
            },
            "Debt to Equity Ratio": {
                "total_liabilities": total_liabilities,
                "shareholders_equity": shareholders_equity,
                "ratio_value": safe_calculate(calculate_debt_to_equity_ratio, total_liabilities, shareholders_equity, "N/A")
            },
            "Gross Margin Ratio": {
                "gross_profit": gross_profit,
                "net_sales": net_sales,
                "ratio_value": safe_calculate(calculate_gross_margin_ratio, gross_profit, net_sales, "N/A")
            },
            "Operating Margin Ratio": {
                "operating_income": operating_income,
                "net_sales": net_sales,
                "ratio_value": safe_calculate(calculate_operating_margin_ratio, operating_income, net_sales, "N/A")
            },
            "Return on Assets Ratio": {
                "net_income": net_income,
                "total_assets": total_assets,
                "ratio_value": safe_calculate(calculate_return_on_assets_ratio, net_income, total_assets, "N/A")
            },
            "Return on Equity Ratio": {
                "net_income": net_income,
                "shareholders_equity": shareholders_equity,
                "ratio_value": safe_calculate(calculate_return_on_equity_ratio, net_income, shareholders_equity, "N/A")
            },
            "Asset Turnover Ratio": {
                "net_sales": net_sales,
                "average_total_assets": average_total_assets,
                "ratio_value": safe_calculate(calculate_asset_turnover_ratio, net_sales, average_total_assets, "N/A")
            },
            "Inventory Turnover Ratio": {
                "cost_of_goods_sold": cost_of_goods_sold,
                "average_inventory": average_inventory,
                "ratio_value": safe_calculate(calculate_inventory_turnover_ratio, cost_of_goods_sold, average_inventory, "N/A")
            },
            "Receivables Turnover Ratio": {
                "net_credit_sales": net_credit_sales,
                "average_accounts_receivable": average_accounts_receivable,
                "ratio_value": safe_calculate(calculate_receivables_turnover_ratio, net_credit_sales, average_accounts_receivable, "N/A")
            },
            "Debt Ratio": {
                "total_liabilities": total_liabilities,
                "total_assets": total_assets,
                "ratio_value": safe_calculate(calculate_debt_ratio, total_liabilities, total_assets, "N/A")
            },
            "Interest Coverage Ratio": {
                "operating_income": operating_income,
                "interest_expenses": interest_expenses,
                "ratio_value": safe_calculate(calculate_interest_coverage_ratio, operating_income, interest_expenses, "N/A")
            }
        }
        
        # Add growth metrics
        if previous_year_sales and net_sales:
            sales_growth = ((net_sales - previous_year_sales) / previous_year_sales) * 100 if previous_year_sales != 0 else "N/A"
            ratios["Sales Growth"] = {
                "current_sales": net_sales,
                "previous_sales": previous_year_sales,
                "growth_percentage": sales_growth
            }
        
        if previous_year_net_income and net_income:
            profit_growth = ((net_income - previous_year_net_income) / previous_year_net_income) * 100 if previous_year_net_income != 0 else "N/A"
            ratios["Profit Growth"] = {
                "current_net_income": net_income,
                "previous_net_income": previous_year_net_income,
                "growth_percentage": profit_growth
            }
        
        if previous_year_total_assets and total_assets:
            asset_growth = ((total_assets - previous_year_total_assets) / previous_year_total_assets) * 100 if previous_year_total_assets != 0 else "N/A"
            ratios["Asset Growth"] = {
                "current_assets": total_assets,
                "previous_assets": previous_year_total_assets,
                "growth_percentage": asset_growth
            }
        
        # Add cash flow metrics
        if operating_cash_flow and net_income:
            cash_to_income_ratio = safe_calculate(lambda x, y: x / y, operating_cash_flow, net_income, "N/A")
            ratios["Cash Flow to Net Income Ratio"] = {
                "operating_cash_flow": operating_cash_flow,
                "net_income": net_income,
                "ratio_value": cash_to_income_ratio
            }
        
        if free_cash_flow is not None:
            ratios["Free Cash Flow"] = {
                "value": free_cash_flow
            }
        
        # Add anomaly detection
        anomalies = []
        
        # Check for suspicious profitability patterns
        if net_income > operating_income and interest_expenses > 0:
            anomalies.append({
                "type": "Income Anomaly",
                "description": "Net income exceeds operating income despite interest expenses",
                "severity": "Medium"
            })
        
        # Check for liquidity concerns
        if isinstance(ratios["Current Ratio"]["ratio_value"], (int, float)) and ratios["Current Ratio"]["ratio_value"] < 1.0:
            anomalies.append({
                "type": "Liquidity Risk",
                "description": f"Current ratio is {ratios['Current Ratio']['ratio_value']:.2f}, below the recommended 1.0 minimum",
                "severity": "High" if ratios["Current Ratio"]["ratio_value"] < 0.8 else "Medium"
            })
        
        # Check for high leverage
        if isinstance(ratios["Debt Ratio"]["ratio_value"], (int, float)) and ratios["Debt Ratio"]["ratio_value"] > 0.7:
            anomalies.append({
                "type": "High Leverage",
                "description": f"Debt ratio is {ratios['Debt Ratio']['ratio_value']:.2f}, indicating high financial leverage",
                "severity": "High" if ratios["Debt Ratio"]["ratio_value"] > 0.8 else "Medium"
            })
        
        # Check for cash flow vs. net income discrepancy
        if operating_cash_flow and net_income:
            if operating_cash_flow < 0 and net_income > 0:
                anomalies.append({
                    "type": "Cash Flow Discrepancy",
                    "description": "Positive net income but negative operating cash flow suggests potential earnings quality issues",
                    "severity": "High"
                })
            elif operating_cash_flow / net_income < 0.5 and net_income > 0:
                anomalies.append({
                    "type": "Cash Flow Discrepancy",
                    "description": f"Operating cash flow is only {(operating_cash_flow/net_income)*100:.0f}% of net income, suggesting potential earnings quality issues",
                    "severity": "Medium"
                })
        
        # Add anomalies to the ratios dictionary
        ratios["Anomalies"] = anomalies
        
        return ratios
    
    def detect_financial_red_flags(self, data: Dict, ratios: Dict) -> Dict:
        """
        Detect potential red flags and anomalies in financial data that could 
        indicate financial distress or manipulation.
        """
        red_flags = []
        
        # Extract relevant data
        income_statement = data.get('income_statement', {})
        balance_sheet = data.get('balance_sheet', {})
        cash_flow = data.get('cash_flow', {})
        
        net_sales = income_statement.get('net_sales')
        previous_year_sales = income_statement.get('previous_year_sales')
        net_income = income_statement.get('net_income')
        operating_income = income_statement.get('operating_income')
        operating_cash_flow = cash_flow.get('operating_cash_flow')
        
        # Red flag: Unusual revenue growth
        if previous_year_sales and net_sales:
            growth_rate = (net_sales - previous_year_sales) / previous_year_sales if previous_year_sales > 0 else 0
            if growth_rate > 0.5:  # 50% growth is unusually high
                red_flags.append({
                    "category": "Revenue",
                    "issue": "Unusually high revenue growth",
                    "details": f"{growth_rate*100:.1f}% increase in revenue from previous year",
                    "severity": "Medium",
                    "recommendation": "Verify revenue recognition policies and major sales events"
                })
        
        # Red flag: Net income higher than operating cash flow (potential earnings management)
        if net_income and operating_cash_flow and net_income > operating_cash_flow * 2:
            red_flags.append({
                "category": "Cash Flow",
                "issue": "Net income significantly exceeds operating cash flow",
                "details": f"Net income is {net_income/operating_cash_flow:.1f}x higher than operating cash flow",
                "severity": "High",
                "recommendation": "Investigate accruals and revenue recognition"
            })
        
        # Red flag: Interest coverage concerns
        interest_coverage = ratios.get('Interest Coverage Ratio', {}).get('ratio_value')
        if isinstance(interest_coverage, (int, float)) and interest_coverage < 2.0:
            red_flags.append({
                "category": "Solvency",
                "issue": "Low interest coverage ratio",
                "details": f"Interest coverage ratio of {interest_coverage:.2f} indicates potential difficulty in meeting interest obligations",
                "severity": "High" if interest_coverage < 1.0 else "Medium",
                "recommendation": "Review debt structure and interest payment capabilities"
            })
        
        # Red flag: Working capital concerns
        current_ratio = ratios.get('Current Ratio', {}).get('ratio_value')
        if isinstance(current_ratio, (int, float)) and current_ratio < 1.0:
            red_flags.append({
                "category": "Liquidity",
                "issue": "Working capital deficiency",
                "details": f"Current ratio of {current_ratio:.2f} indicates insufficient short-term assets to cover short-term liabilities",
                "severity": "High" if current_ratio < 0.8 else "Medium",
                "recommendation": "Address short-term liquidity through debt restructuring or additional financing"
            })
        
        # Return the list of red flags
        return {
            "red_flags": red_flags,
            "has_critical_issues": any(flag["severity"] == "High" for flag in red_flags),
            "has_concerns": len(red_flags) > 0
        }
    
    def analyze_financial_document(self, document_content: str) -> Dict[str, Any]:
        """
        Analyze a financial document and return extracted data and calculated ratios
        
        Args:
            document_content: The text content of the financial document
            
        Returns:
            Dict with extracted financial data and calculated ratios
        """
        try:
            # Validate input
            if not document_content or len(document_content.strip()) == 0:
                raise ValueError("Document content is empty")
                
            logging.info(f"Starting financial document analysis. Document length: {len(document_content)} characters")
            start_time = time.time()
            
            # Step 1: Extract financial data
            logging.info("Step 1: Extracting financial data...")
            extraction_result = self.extraction_chain.invoke({"document_content": document_content})
            
            logging.info(f"Raw extraction result: {extraction_result['extracted_data'][:200]}...")
            
            # Convert the string JSON to a Python dict
            try:
                # First try direct JSON parsing
                extracted_data = json.loads(extraction_result["extracted_data"])
                logging.info("Successfully parsed JSON directly")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response text
                logging.info("Direct JSON parsing failed, trying to extract JSON from text")
                json_match = re.search(r'```json\s*(.*?)\s*```', extraction_result["extracted_data"], re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group(1))
                        logging.info("Successfully extracted and parsed JSON from text")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error after extraction: {e}")
                        raise ValueError(f"Invalid JSON format after extraction: {e}")
                else:
                    # If no JSON block found, provide a fallback empty structure
                    logging.error("No valid JSON found in extraction result")
                    extracted_data = {
                        "company_name": "",
                        "reporting_period": "",
                        "currency": "",
                        "industry": "",
                        "income_statement": {},
                        "balance_sheet": {},
                        "cash_flow": {},
                        "notes": {
                            "adj_ebitda_available": False,
                            "adj_ebitda_details": "",
                            "adj_working_capital_available": False,
                            "adj_working_capital_details": "",
                            "risk_factors": [],
                            "significant_events": []
                        }
                    }
                    
            # Step 2: Calculate financial ratios
            logging.info("Step 2: Calculating financial ratios...")
            calculated_ratios = self.calculate_financial_ratios(extracted_data)
            
            # Step 3: Detect financial red flags and anomalies
            logging.info("Step 3: Detecting financial red flags and anomalies...")
            red_flags = self.detect_financial_red_flags(extracted_data, calculated_ratios)
            
            # Step 4: Generate business overview and key findings
            logging.info("Step 4: Generating business overview and key findings...")
            business_overview = self.generate_business_overview(extracted_data)
            key_findings = self.generate_key_findings(extracted_data, calculated_ratios)
            
            end_time = time.time()
            logging.info(f"Financial analysis completed in {end_time - start_time:.2f} seconds")
            
            # Return combined result
            return {
                "extracted_data": extracted_data,
                "calculated_ratios": calculated_ratios,
                "red_flags": red_flags,
                "business_overview": business_overview,
                "key_findings": key_findings
            }
            
        except Exception as e:
            logging.exception("Error in analyze_financial_document")
            # Return error information
            return {
                "error": str(e),
                "extracted_data": {},
                "calculated_ratios": {},
                "red_flags": {},
                "business_overview": "Error analyzing document: " + str(e),
                "key_findings": "Error analyzing document: " + str(e)
            }
            
    def generate_business_overview(self, extracted_data: Dict) -> str:
        """
        Generate a concise business overview based on the extracted data.
        """
        try:
            result = self.overview_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2)
            })
            return result["business_overview"].strip()
        except Exception as e:
            logging.exception("Error generating business overview")
            return f"Error generating business overview: {str(e)}"
        
    def generate_key_findings(self, extracted_data: Dict, calculated_ratios: Dict) -> Dict:
        """
        Generate key findings and insights based on the extracted data and calculated ratios.
        Returns a JSON dict with keys: 'key_findings', 'red_flags', 'sentiment_analysis', and 'business_model'.
        """
        try:
            # If there are benchmarks and anomalies, add them to extracted_data.
            industry = extracted_data.get("industry", "")
            industry_benchmarks = {
                "Retail": {"Current Ratio": 1.5, "Gross Margin Ratio": 0.25, "Inventory Turnover Ratio": 4.0, "Debt Ratio": 0.5},
                "Manufacturing": {"Current Ratio": 1.8, "Gross Margin Ratio": 0.35, "Inventory Turnover Ratio": 5.0, "Debt Ratio": 0.45},
                "Technology": {"Current Ratio": 2.5, "Gross Margin Ratio": 0.60, "Inventory Turnover Ratio": 10.0, "Debt Ratio": 0.35},
                "Financial": {"Current Ratio": 1.1, "Return on Assets Ratio": 0.01, "Debt Ratio": 0.85, "Return on Equity Ratio": 0.12}
            }
            if industry and industry in industry_benchmarks:
                benchmark_data = []
                for ratio_name, benchmark_value in industry_benchmarks[industry].items():
                    ratio_data = calculated_ratios.get(ratio_name, {})
                    actual_value = ratio_data.get("ratio_value", "N/A")
                    if isinstance(actual_value, (int, float)) and isinstance(benchmark_value, (int, float)):
                        comparison = "above" if actual_value > benchmark_value else "below"
                        benchmark_data.append({
                            "ratio": ratio_name,
                            "actual": actual_value,
                            "benchmark": benchmark_value,
                            "comparison": comparison
                        })
                if benchmark_data:
                    extracted_data["industry_benchmarks"] = benchmark_data

            if "Anomalies" in calculated_ratios and calculated_ratios["Anomalies"]:
                extracted_data["anomalies"] = calculated_ratios["Anomalies"]

            result = self.findings_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2),
                "calculated_ratios": json.dumps(calculated_ratios, indent=2)
            })
            parsed = json.loads(result["key_findings"])
            key_findings = parsed.get("key_findings", "")
            red_flags = parsed.get("red_flags", [])
            # NEW: Obtain sentiment analysis and business model recommendations
            sentiment_result = self.sentiment_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2)
            })
            business_model_result = self.business_model_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2)
            })
            return {
                "key_findings": key_findings,
                "red_flags": red_flags,
                "sentiment_analysis": sentiment_result.get("sentiment_analysis", "").strip(),
                "business_model": business_model_result.get("business_model", "").strip()
            }
        except Exception as e:
            logging.exception("Error generating key findings")
            return {
                "key_findings": f"Error generating key findings: {str(e)}",
                "red_flags": [],
                "sentiment_analysis": "N/A",
                "business_model": "N/A"
            }