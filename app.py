from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import json
import sys
import os
import base64

# Import your analytics module
try:
    from analytics_module import (
        UtilizationAnalyticsEngine,
        VALID_GROUPBY_COLUMNS
    )
except ImportError as e:
    print(f"Import error: {e}")

app = Flask(__name__)

# Initialize engine once (Cloud Functions reuses containers)
engine = None

def initialize_engine():
    global engine
    if engine is None:
        try:
            API_KEY = "AIzaSyAaK6AM5aEa4LsNfC7h40YrPnLchgymqYs"
            DATA_PATHS = {
                'prism': "gs://basedataadmin/data_for_model/prism_df.csv",
                'subcon': "gs://basedataadmin/data_for_model/subcon_df.csv",
                'mapping': "gs://basedataadmin/data_for_model/project_mapping_df.csv",
                'forecast': "gs://basedataadmin/data_for_model/forecast_df.csv"
            }
            engine = UtilizationAnalyticsEngine(API_KEY, DATA_PATHS)
            return True
        except Exception as e:
            print(f"Failed to initialize engine: {e}")
            return False
    return True

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Util Agent - Analytics Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #000149;
        }
        .header h1 {
            color: #000149;
            margin: 0;
        }
        .query-section {
            margin-bottom: 30px;
        }
        .query-input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .submit-btn {
            background: #000149;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .submit-btn:hover {
            background: #2A788E;
        }
        .result-section {
            margin-top: 20px;
        }
        .metric-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #000149;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #000149;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .table-container {
            overflow-x: auto;
            margin: 15px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        th {
            background: #000149;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .insight-section, .recommendation-section {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .example-queries {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        .example-query {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            cursor: pointer;
            border: 1px solid #ddd;
        }
        .example-query:hover {
            background: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Util Agent 📊</h1>
            <div>Your intelligent utilization analytics assistant</div>
        </div>

        <div class="query-section">
            <h3>Ask me anything about utilization analytics:</h3>
            <form id="queryForm">
                <input type="text" name="query" class="query-input" 
                       placeholder="e.g., Analyze EDL Garrett, Show me projects with NBL > 25%, Give me utilization by all EDLs..."
                       required>
                <button type="submit" class="submit-btn">Analyze</button>
            </form>

            <div class="example-queries">
                <div class="example-query" onclick="setQuery('Give me utilization by all EDLs')">
                    Give me utilization by all EDLs
                </div>
                <div class="example-query" onclick="setQuery('Analyze EDL Garrett')">
                    Analyze EDL Garrett
                </div>
                <div class="example-query" onclick="setQuery('Show me projects with NBL > 25%')">
                    Show me projects with NBL > 25%
                </div>
                <div class="example-query" onclick="setQuery('What\\'s the TL unbilled impact on Google portfolio?')">
                    What's the TL unbilled impact?
                </div>
                <div class="example-query" onclick="setQuery('Compare EDL Bharath vs EDL Pradeepan')">
                    Compare EDL Bharath vs EDL Pradeepan
                </div>
            </div>
        </div>

        <div id="resultSection" class="result-section" style="display: none;">
            <!-- Results will be populated here by JavaScript -->
        </div>

        <div id="loadingSection" class="loading" style="display: none;">
            <h3>Analyzing your query...</h3>
            <p>Please wait while we process your request.</p>
        </div>
    </div>

    <script>
        function setQuery(query) {
            document.querySelector('.query-input').value = query;
        }

        document.getElementById('queryForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const query = document.querySelector('.query-input').value;
            
            // Show loading, hide results
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';
            
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loadingSection').style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loadingSection').style.display = 'none';
                displayError('An error occurred: ' + error.message);
            });
        });

        function displayResults(data) {
            const resultSection = document.getElementById('resultSection');
            resultSection.style.display = 'block';
            
            let html = '';
            
            if (data.error) {
                html = `<div class="error">${data.error}</div>`;
            } else {
                // Narrative
                if (data.narrative) {
                    html += `<div style="background: #e8f5e8; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                                <h3>Analysis Summary</h3>
                                <p>${data.narrative}</p>
                            </div>`;
                }
                
                // Metrics
                if (data.metrics && Object.keys(data.metrics).length > 0) {
                    html += '<h3>Key Metrics</h3><div class="metric-cards">';
                    for (const [key, value] of Object.entries(data.metrics)) {
                        html += `<div class="metric-card">
                                    <div class="metric-label">${key}</div>
                                    <div class="metric-value">${value}</div>
                                </div>`;
                    }
                    html += '</div>';
                }
                
                // Tables
                if (data.tables && Object.keys(data.tables).length > 0) {
                    for (const [tableName, tableData] of Object.entries(data.tables)) {
                        html += `<h3>${tableName}</h3><div class="table-container">`;
                        if (tableData.length > 0) {
                            const headers = Object.keys(tableData[0]);
                            html += '<table><thead><tr>';
                            headers.forEach(header => {
                                html += `<th>${header}</th>`;
                            });
                            html += '</tr></thead><tbody>';
                            
                            tableData.forEach(row => {
                                html += '<tr>';
                                headers.forEach(header => {
                                    html += `<td>${row[header] || ''}</td>`;
                                });
                                html += '</tr>';
                            });
                            
                            html += '</tbody></table>';
                        }
                        html += '</div>';
                    }
                }
                
                // Insights
                if (data.insights && data.insights.length > 0) {
                    html += '<div class="insight-section"><h3>💡 Key Insights</h3><ul>';
                    data.insights.forEach(insight => {
                        html += `<li>${insight}</li>`;
                    });
                    html += '</ul></div>';
                }
                
                // Recommendations
                if (data.recommendations && data.recommendations.length > 0) {
                    html += '<div class="recommendation-section"><h3>🎯 Recommendations</h3><ul>';
                    data.recommendations.forEach(rec => {
                        html += `<li>${rec}</li>`;
                    });
                    html += '</ul></div>';
                }
            }
            
            resultSection.innerHTML = html;
        }

        function displayError(message) {
            const resultSection = document.getElementById('resultSection');
            resultSection.style.display = 'block';
            resultSection.innerHTML = `<div class="error">${message}</div>`;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "No query provided"})
        
        # Initialize engine if not already done
        if not initialize_engine():
            return jsonify({"error": "Failed to initialize analytics engine"})
        
        # Process the query
        result = engine.process_query(query)
        
        # Format the response for the web interface
        formatted_response = format_response(result)
        
        return jsonify(formatted_response)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

# Copy the format_response function and its helpers from your streamlit_app.py
def format_response(result: dict) -> dict:
    """Format the engine result into structured response for display"""
    
    formatted = {
        "narrative": "",
        "tables": {},
        "metrics": {},
        "insights": [],
        "recommendations": []
    }
    
    # Handle error
    if "error" in result:
        formatted["error"] = result["error"]
        return formatted
    
    # Extract narrative
    if "narrative_summary" in result:
        formatted["narrative"] = result["narrative_summary"]
    
    intent = result.get("query_intent", "")
    
    # Format based on intent type
    if intent == "edl_analysis":
        formatted = format_edl_analysis(result, formatted)
    elif intent == "target_achievement":
        formatted = format_target_achievement(result, formatted)
    elif intent == "resource_optimization":
        formatted = format_resource_optimization(result, formatted)
    elif intent == "unbilled_analysis":
        formatted = format_unbilled_analysis(result, formatted)
    elif intent == "buffer_analysis":
        formatted = format_buffer_analysis(result, formatted)
    elif intent == "diagnostic_analysis":
        formatted = format_diagnostic_analysis(result, formatted)
    elif intent == "nbl_threshold_analysis":
        formatted = format_nbl_threshold(result, formatted)
    elif intent == "comparative_analysis":
        formatted = format_comparative_analysis(result, formatted)
    elif intent == "project_distribution":
        formatted = format_project_distribution(result, formatted)
    elif intent == "location_breakdown":
        formatted = format_location_breakdown(result, formatted)
    elif intent == "tl_impact_analysis":
        formatted = format_tl_impact(result, formatted)
    else:
        # Default metric fetch
        formatted = format_metric_fetch(result, formatted)
    
    return formatted

# Copy all the format helper functions from your streamlit_app.py
def format_edl_analysis(result: dict, formatted: dict) -> dict:
    """Format EDL analysis results"""
    
    # Check if it's all EDLs or single EDL
    if result.get('analysis_type') == 'all_edls_comparison':
        formatted["narrative"] = result.get("narrative_summary", "EDL Analysis Complete")
        
        # Complete metrics table
        if result.get('complete_metrics_table'):
            formatted["tables"]["EDL Metrics Overview"] = result['complete_metrics_table']
        
        # Comparison metrics
        if result.get('comparison_metrics'):
            comp = result['comparison_metrics']
            formatted["metrics"]["Average Utilization"] = f"{comp.get('average_utilization', 0):.1f}%"
            
            if comp.get('ranking'):
                top_edls = comp['ranking'][:5]
                formatted["tables"]["Top Performing EDLs"] = top_edls
    
    else:
        # Single EDL analysis
        if result.get('summary'):
            summary = result['summary']
            formatted["metrics"]["Overall Utilization"] = f"{summary.get('overall_utilization', 0)}%"
            formatted["metrics"]["Total FTE"] = f"{summary.get('total_fte', 0):.1f}"
            formatted["metrics"]["Total Projects"] = summary.get('total_projects', 0)
            formatted["metrics"]["Performance Tier"] = summary.get('performance_tier', 'Unknown')
        
        # Complete metrics table
        if result.get('complete_metrics_table'):
            formatted["tables"]["Portfolio Metrics"] = result['complete_metrics_table']
        
        # Key insights
        if result.get('key_insights'):
            formatted["insights"] = result['key_insights']
    
    return formatted

# Copy all other format functions from your streamlit_app.py...
# [Include all the format_* functions from your original streamlit_app.py here]

def format_target_achievement(result: dict, formatted: dict) -> dict:
    """Format target achievement analysis"""
    
    formatted["metrics"]["Current Utilization"] = f"{result.get('current_utilization', 0)}%"
    formatted["metrics"]["Target Utilization"] = f"{result.get('target_utilization', 0)}%"
    formatted["metrics"]["Gap (FTE)"] = f"{result.get('utilization_gap_fte', 0):.1f}"
    formatted["metrics"]["Optimizable NBL"] = f"{result.get('optimizable_nbl_fte', 0):.1f}"
    
    if result.get('top_projects'):
        formatted["tables"]["Top Projects for Improvement"] = pd.DataFrame(result['top_projects'])
    
    if result.get('target_achievable'):
        formatted["insights"].append("✅ Target is achievable with current optimizable resources")
    else:
        formatted["insights"].append("⚠️ Additional strategies needed to achieve target")
    
    return formatted

def format_resource_optimization(result: dict, formatted: dict) -> dict:
    """Format resource optimization results"""
    
    formatted["metrics"]["Current Utilization"] = f"{result.get('current_utilization', 0)}%"
    formatted["metrics"]["New Utilization"] = f"{result.get('new_utilization', 0)}%"
    formatted["metrics"]["Improvement"] = f"+{result.get('utilization_improvement', 0):.1f}%"
    formatted["metrics"]["Total Removed"] = f"{result.get('total_removed', 0)} FTE"
    
    if result.get('removal_plan'):
        formatted["tables"]["Removal Plan"] = pd.DataFrame(result['removal_plan'])
    
    if result.get('unbilled_breakdown'):
        formatted["tables"]["Unbilled Breakdown"] = pd.DataFrame(
            list(result['unbilled_breakdown'].items()),
            columns=['Category', 'FTE']
        )
    
    return formatted

def format_unbilled_analysis(result: dict, formatted: dict) -> dict:
    """Format unbilled analysis results"""
    
    formatted["metrics"]["Total Unbilled FTE"] = f"{result.get('total_unbilled_fte', 0):.1f}"
    formatted["metrics"]["Total Associates"] = result.get('total_unbilled_associates', 0)
    
    if result.get('strategic_breakdown'):
        strategic_df = pd.DataFrame([
            {
                "Category": k,
                "Unbilled FTE": v.get('Unbilled_FTE', 0),
                "Associate Count": v.get('associate_count', 0)
            }
            for k, v in result['strategic_breakdown'].items()
        ])
        formatted["tables"]["Strategic Breakdown"] = strategic_df
    
    if result.get('detailed_category_breakdown'):
        category_df = pd.DataFrame([
            {
                "Category": k,
                "Unbilled FTE": v.get('Unbilled_FTE', 0),
                "Associate Count": v.get('associate_count', 0)
            }
            for k, v in result['detailed_category_breakdown'].items()
        ])
        formatted["tables"]["Detailed Category Breakdown"] = category_df
    
    if result.get('msa_vs_non_msa'):
        msa_data = result['msa_vs_non_msa']
        formatted["metrics"]["MSA Buffer"] = f"{msa_data.get('msa_buffer_fte', 0):.1f} FTE"
        formatted["metrics"]["Non-MSA Buffer"] = f"{msa_data.get('non_msa_buffer_fte', 0):.1f} FTE"
    
    return formatted

def format_buffer_analysis(result: dict, formatted: dict) -> dict:
    """Format buffer analysis results"""
    
    formatted["metrics"]["MSA Buffer"] = f"{result.get('msa_buffer_fte', 0):.1f} FTE"
    formatted["metrics"]["Non-MSA Buffer"] = f"{result.get('non_msa_buffer_fte', 0):.1f} FTE"
    formatted["metrics"]["Total Movable"] = f"{result.get('total_movable_fte', 0):.1f} FTE"
    formatted["metrics"]["Optimization Opportunity"] = f"{result.get('optimization_opportunity', 0):.1f} FTE"
    
    if result.get('detailed_breakdown', {}).get('category_breakdown'):
        category_df = pd.DataFrame([
            {
                "Category": k,
                "Unbilled FTE": v.get('Unbilled_FTE', 0),
                "Associate Count": v.get('associate_count', 0)
            }
            for k, v in result['detailed_breakdown']['category_breakdown'].items()
        ])
        formatted["tables"]["Buffer Category Breakdown"] = category_df
    
    return formatted

def format_diagnostic_analysis(result: dict, formatted: dict) -> dict:
    """Format diagnostic analysis results"""
    
    formatted["metrics"]["Projects Analyzed"] = result.get('projects_analyzed', 0)
    formatted["metrics"]["Average Utilization"] = f"{result.get('average_utilization', 0)}%"
    
    if result.get('insights'):
        formatted["insights"] = result['insights'][:5]
    
    if result.get('recommendations'):
        formatted["recommendations"] = result['recommendations'][:5]
    
    return formatted

def format_nbl_threshold(result: dict, formatted: dict) -> dict:
    """Format NBL threshold analysis"""
    
    formatted["metrics"]["Threshold"] = f"{result.get('threshold', 0)}%"
    formatted["metrics"]["Projects Above Threshold"] = result.get('projects_above_threshold', 0)
    formatted["metrics"]["High NBL FTE"] = f"{result.get('total_high_nbl_fte', 0):.1f}"
    formatted["metrics"]["Share of Portfolio"] = f"{result.get('high_nbl_share_%', 0):.1f}%"
    
    if result.get('high_nbl_projects'):
        formatted["tables"]["High NBL Projects"] = pd.DataFrame(result['high_nbl_projects'])
    
    if result.get('recommendations'):
        formatted["recommendations"] = result['recommendations']
    
    return formatted

def format_comparative_analysis(result: dict, formatted: dict) -> dict:
    """Format comparative analysis results"""
    
    if result.get('comparison_data'):
        formatted["tables"]["EDL Comparison"] = pd.DataFrame(result['comparison_data'])
    
    return formatted

def format_project_distribution(result: dict, formatted: dict) -> dict:
    """Format project distribution analysis"""
    
    formatted["metrics"]["Total Projects"] = result.get('total_projects', 0)
    formatted["metrics"]["Total FTE"] = f"{result.get('total_fte', 0):.1f}"
    
    if result.get('project_distribution'):
        dist_df = pd.DataFrame([
            {
                "Project": k,
                **v
            }
            for k, v in result['project_distribution'].items()
        ])
        formatted["tables"]["Project Distribution"] = dist_df
    
    if result.get('concentration_metrics'):
        conc = result['concentration_metrics']
        formatted["metrics"]["Top Project Share"] = f"{conc.get('top_project_share_%', 0):.1f}%"
    
    return formatted

def format_location_breakdown(result: dict, formatted: dict) -> dict:
    """Format location breakdown analysis"""
    
    formatted["metrics"]["Total Locations"] = result.get('total_locations', 0)
    
    if result.get('location_breakdown'):
        loc_df = pd.DataFrame([
            {
                "Location": k,
                **v
            }
            for k, v in result['location_breakdown'].items()
        ])
        formatted["tables"]["Location Breakdown"] = loc_df
    
    if result.get('performance_metrics'):
        perf = result['performance_metrics']
        formatted["metrics"]["Utilization Std Dev"] = f"{perf.get('utilization_std', 0):.1f}%"
        formatted["metrics"]["Utilization Range"] = f"{perf.get('utilization_range', 0):.1f}%"
    
    return formatted

def format_tl_impact(result: dict, formatted: dict) -> dict:
    """Format TL impact analysis"""
    
    formatted["metrics"]["Total TL FTE"] = f"{result.get('total_tl_fte', 0):.1f}"
    formatted["metrics"]["TL Unbilled FTE"] = f"{result.get('total_tl_unbilled_fte', 0):.1f}"
    formatted["metrics"]["TL Unbilled Impact"] = f"{result.get('tl_unbilled_impact_%', 0):.1f}%"
    formatted["metrics"]["TL Share of Unbilled"] = f"{result.get('tl_share_of_unbilled_%', 0):.1f}%"
    
    if result.get('tl_unbilled_by_category'):
        cat_df = pd.DataFrame(
            list(result['tl_unbilled_by_category'].items()),
            columns=['Category', 'FTE']
        )
        formatted["tables"]["TL Unbilled by Category"] = cat_df
    
    if result.get('critical_insights'):
        formatted["insights"] = result['critical_insights']
    
    return formatted

def format_metric_fetch(result: dict, formatted: dict) -> dict:
    """Format default metric fetch results"""
    
    if result.get('metrics_summary'):
        summary = result['metrics_summary']
        formatted["metrics"]["Average Utilization"] = f"{summary.get('average_utilization', 0)}%"
        formatted["metrics"]["Average NBL"] = f"{summary.get('average_nbl', 0)}%"
        formatted["metrics"]["Total Groups"] = summary.get('total_groups', 0)
        
        if summary.get('top_performers'):
            formatted["tables"]["Top Performers"] = pd.DataFrame(summary['top_performers'])
        
        if summary.get('concern_areas'):
            formatted["tables"]["Areas of Concern"] = pd.DataFrame(summary['concern_areas'])
    
    if result.get('raw_metrics'):
        formatted["tables"]["Complete Metrics"] = pd.DataFrame(result['raw_metrics'])
    
    return formatted

def _display_metrics(metrics: dict):
    """Safely render the metric cards."""
    if not metrics:
        return

    n = len(metrics)
    if n == 0:
        return
    elif n == 1:
        name, value = next(iter(metrics.items()))
        st.metric(label=name, value=value)
    else:
        cols = st.columns(n)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(label=name, value=value)

def get_base64_encoded_image(image_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# [Copy all remaining format functions...]

if __name__ == '__main__':
    app.run(debug=True)