import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import re
from typing import Dict, Any, List, Tuple

# === VALID GROUPING COLUMNS ===
VALID_GROUPBY_COLUMNS = [
    "DBO_Grade",
    "Project_Billability",
    "Utilization_Location",
    "Project_Name",
    "Is_Onsite",
    "Rev_Mapping",
    "EDL_Name"
]

# === STRATEGIC UNBILLED CATEGORIZATION ===
MOVABLE_CATEGORIES = [
    "Unbilled - Non MSA Buffer",
    "Unbilled - KT"
]

UNMOVABLE_CATEGORIES = [
    "Unbilled - MSA buffer",
    "One % alloc for T&E/Prj access",
    "Allocated - Awaiting billing",
    "Unbilled - Management",  # Should be billable but not to be moved
    "Billed"  # Already utilized
]

class LLMFirstQueryParsingAgent:
    def __init__(self, api_key: str, mapping_file_path: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        try:
            self.mapping_df = pd.read_csv(mapping_file_path)
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            self.mapping_df = pd.DataFrame()
        self.project_context = self._build_llm_context()

    def _build_llm_context(self) -> str:
        if self.mapping_df.empty:
            return "No project mapping data available."

        context = "Available Projects and Hierarchies:\n\n"
        try:
            rev_mapping_groups = self.mapping_df.groupby('Rev_Mapping')
            for rev_mapping, group in rev_mapping_groups:
                context += f"REV_MAPPING: {rev_mapping}\n"
                context += f"EDL: {group['EDL_Name'].iloc[0]}\n"
                context += "Projects:\n"
                for _, row in group.iterrows():
                    context += f" - {row['Project_Name']} (ID: {row['Project_ID']})\n"
                context += "\n"
        except Exception as e:
            print(f"Error building context: {e}")
        return context

    def create_comprehensive_prompt(self, query: str) -> str:
        prompt = f"""
        You are an advanced utilization analytics query parser. Parse this query: "{query}"

        {self.project_context}

        Analyze the query and return JSON with:

        1. intent: "target_achievement", "resource_optimization", "diagnostic_analysis", "metric_fetch", "ranking", "edl_analysis", "unbilled_analysis", "buffer_analysis", "project_distribution", "location_breakdown", "tl_impact_analysis", "nbl_threshold_analysis", "comparative_analysis"
        2. entities: rev_mappings, edl_names, project_names mentioned
        3. parameters: nbl_count (if removal query), target_scope (portfolio/edl/project), nbl_threshold (for NBL > X% queries)
        4. focus_metrics: ["overall_utilization", "nbl_utilization", "nbl_tl_utilization"]
        5. analysis_type: "msa_analysis", "non_msa_optimization", "unbilled_breakdown", "project_distribution", "location_breakdown", "tl_impact"
        6. group_by: if "by X" or "for X", set group_by = X (must be one of: DBO_Grade, Project_Billability, Utilization Location, Project_Name, Is_Onsite, Rev_Mapping, EDL_Name)

        NEW Intent Classification Examples:
        - "Analyze EDL Bharath's project distribution" → "project_distribution"
        - "What's the performance breakdown for EDL Garrett by location?" → "location_breakdown"
        - "What's the TL unbilled impact on Google portfolio?" → "tl_impact_analysis"
        - "Show me projects with NBL > 25%" → "nbl_threshold_analysis" with nbl_threshold=25
        - "Compare EDL Bharath vs EDL Pradeepan" → "comparative_analysis"
        - "Show me MSA vs Non-MSA buffers for Google Street View" → "buffer_analysis" with detailed categorization
        - "Give me unbilled analysis for Google Street View - Drive Ops" → "unbilled_analysis" with detailed categorization
        
        NOTE : Portfolio, portfolio also means rev_mappings
        Return ONLY valid JSON.
        """
        return prompt

    def parse_query_with_llm(self, query: str) -> Dict[str, Any]:
        try:
            prompt = self.create_comprehensive_prompt(query)
            response = self.model.generate_content(prompt)
            response_text = self._extract_json_from_response(response.text)
            parsed_result = json.loads(response_text)
            return self._validate_llm_output(parsed_result)
        except Exception as e:
            print(f"LLM parsing error: {e}")
            return self._create_error_response(str(e))

    def _extract_json_from_response(self, response_text: str) -> str:
        try:
            response_text = response_text.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json_match.group()
            return response_text
        except Exception as e:
            print(f"JSON extraction error: {e}")
            return '{}'

    def _validate_llm_output(self, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = {
            "intent": "metric_fetch",
            "entities": {"rev_mappings": [], "edl_names": [], "project_names": []},
            "parameters": {},
            "focus_metrics": [],
            "analysis_type": "unbilled_breakdown",
            "group_by": "Rev_Mapping"
        }
        validated = self._deep_merge(required_fields, llm_output)

        # Ensure parameters has nbl_count with default value 0
        if "parameters" not in validated:
            validated["parameters"] = {}
        if "nbl_count" not in validated["parameters"]:
            validated["parameters"]["nbl_count"] = 0

        return validated

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        result = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "intent": "metric_fetch",
            "entities": {"rev_mappings": [], "edl_names": [], "project_names": []},
            "parameters": {"nbl_count": 0},
            "focus_metrics": [],
            "analysis_type": "unbilled_breakdown",
            "group_by": "Rev_Mapping"
        }

class MetricCalculationAgent:
    def __init__(self):
        self.tl_plus_grades = ['GM', 'SDM', 'DGM', 'TM', 'Cont', 'SD', 'AVP', 'VP', 'Dir']

    def calculate_all_metrics(self, prism_data: pd.DataFrame, subcon_data: pd.DataFrame,
                            group_by: str = "Rev_Mapping") -> pd.DataFrame:
        try:
            if prism_data.empty:
                print("No PRISM data available")
                return pd.DataFrame()

            print(f"Calculating metrics for {len(prism_data)} rows, grouped by: {group_by}")

            # Create a copy to avoid SettingWithCopyWarning
            prism_data_clean = prism_data.copy()

            # Ensure numeric columns on the copy
            numeric_cols = ['Billed_FTE', 'Total_FTE', 'Unbilled_FTE']
            for col in numeric_cols:
                if col in prism_data_clean.columns:
                    prism_data_clean[col] = pd.to_numeric(prism_data_clean[col], errors='coerce').fillna(0)

            # Use the cleaned copy for all calculations
            overall_util = prism_data_clean.groupby(group_by).agg({
                'Billed_FTE': 'sum',
                'Total_FTE': 'sum'
            }).reset_index()
            overall_util['overall_utilization_%'] = np.where(
                overall_util['Total_FTE'] > 0,
                (overall_util['Billed_FTE'] / overall_util['Total_FTE']) * 100,
                0
            )

            unbilled_data = prism_data_clean[prism_data_clean['Status'] == 'Unbilled'].groupby(group_by)['Unbilled_FTE'].sum().reset_index()
            unbilled_data.columns = [group_by, 'total_unbilled_fte']

            # Enhanced: Calculate associate counts for unbilled
            unbilled_associate_counts = prism_data_clean[prism_data_clean['Status'] == 'Unbilled'].groupby(group_by).size().reset_index()
            unbilled_associate_counts.columns = [group_by, 'unbilled_associate_count']

            total_capacity = prism_data_clean.groupby(group_by)['Total_FTE'].sum().reset_index()
            if not subcon_data.empty and 'Rev_Mapping' in subcon_data.columns:
                subcon_agg = subcon_data.groupby('Rev_Mapping')['subcon'].sum().reset_index()
                total_capacity = total_capacity.merge(subcon_agg, left_on=group_by, right_on='Rev_Mapping', how='left')
                total_capacity['subcon_fte'] = total_capacity['subcon'].fillna(0)
                total_capacity['total_capacity'] = total_capacity['Total_FTE'] + total_capacity['subcon_fte']
            else:
                total_capacity['total_capacity'] = total_capacity['Total_FTE']
                total_capacity['subcon_fte'] = 0

            tl_unbilled = prism_data_clean[
                (prism_data_clean['Status'] == 'Unbilled') &
                (prism_data_clean['DBO_Grade'].isin(self.tl_plus_grades))
            ].groupby(group_by)['Unbilled_FTE'].sum().reset_index()
            tl_unbilled.columns = [group_by, 'unbilled_tl_fte']

            # Enhanced: Strategic unbilled breakdown with associate counts
            unbilled_breakdown = self._calculate_strategic_unbilled_breakdown(prism_data_clean, group_by)

            metrics_df = overall_util.merge(unbilled_data, on=group_by, how='left')
            metrics_df = metrics_df.merge(unbilled_associate_counts, on=group_by, how='left')
            metrics_df = metrics_df.merge(total_capacity[[group_by, 'total_capacity', 'subcon_fte']], on=group_by, how='left')
            metrics_df = metrics_df.merge(tl_unbilled, on=group_by, how='left')

            metrics_df['total_unbilled_fte'] = metrics_df['total_unbilled_fte'].fillna(0)
            metrics_df['unbilled_tl_fte'] = metrics_df['unbilled_tl_fte'].fillna(0)
            metrics_df['unbilled_associate_count'] = metrics_df['unbilled_associate_count'].fillna(0)

            metrics_df['nbl_utilization_%'] = np.where(
                metrics_df['total_capacity'] > 0,
                (metrics_df['total_unbilled_fte'] / metrics_df['total_capacity']) * 100,
                0
            )

            metrics_df['nbl_tl_utilization_%'] = np.where(
                metrics_df['Total_FTE'] > 0,
                (metrics_df['unbilled_tl_fte'] / metrics_df['Total_FTE']) * 100,
                0
            )

            metrics_df['nbl_tl_w_subcon_%'] = np.where(
                metrics_df['total_capacity'] > 0,
                (metrics_df['unbilled_tl_fte'] / metrics_df['total_capacity']) * 100,
                0
            )

            if not unbilled_breakdown.empty:
                metrics_df = metrics_df.merge(unbilled_breakdown, on=group_by, how='left')

            print(f"Calculated metrics for {len(metrics_df)} groups")
            return metrics_df

        except Exception as e:
            print(f"Metric calculation error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _calculate_strategic_unbilled_breakdown(self, prism_data: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """Calculate strategic unbilled breakdown with FTE and associate counts"""
        try:
            unbilled_data = prism_data[prism_data['Status'] == 'Unbilled'].copy()

            if unbilled_data.empty:
                return pd.DataFrame()

            # Calculate FTE by category
            fte_breakdown = unbilled_data.groupby([group_by, 'assignstatetagdesc'])['Unbilled_FTE'].sum().unstack(fill_value=0).reset_index()

            # Calculate associate counts by category
            count_breakdown = unbilled_data.groupby([group_by, 'assignstatetagdesc']).size().unstack(fill_value=0).reset_index()
            count_breakdown = count_breakdown.add_suffix('_count')
            count_breakdown = count_breakdown.rename(columns={f'{group_by}_count': group_by})

            # Merge FTE and count breakdowns
            breakdown = fte_breakdown.merge(count_breakdown, on=group_by, how='left')

            # Calculate strategic totals
            movable_columns = [col for col in fte_breakdown.columns if col in MOVABLE_CATEGORIES]
            unmovable_columns = [col for col in fte_breakdown.columns if col in UNMOVABLE_CATEGORIES]

            if movable_columns:
                breakdown['movable_unbilled_fte'] = breakdown[movable_columns].sum(axis=1)
                breakdown['movable_associate_count'] = breakdown[[f'{col}_count' for col in movable_columns]].sum(axis=1)

            if unmovable_columns:
                breakdown['unmovable_unbilled_fte'] = breakdown[unmovable_columns].sum(axis=1)
                breakdown['unmovable_associate_count'] = breakdown[[f'{col}_count' for col in unmovable_columns]].sum(axis=1)

            return breakdown

        except Exception as e:
            print(f"Strategic unbilled breakdown error: {e}")
            return pd.DataFrame()

    def analyze_tl_unbilled_impact(self, prism_data: pd.DataFrame, scope: str = "portfolio") -> Dict[str, Any]:
        """Analyze TL unbilled impact on portfolio/EDL/project"""
        try:
            # Filter TL+ grades
            tl_data = prism_data[prism_data['DBO_Grade'].isin(self.tl_plus_grades)].copy()

            if tl_data.empty:
                return {"error": "No TL+ grade data available"}

            # TL unbilled analysis
            tl_unbilled = tl_data[tl_data['Status'] == 'Unbilled']
            total_tl_unbilled_fte = tl_unbilled['Unbilled_FTE'].sum()
            total_tl_fte = tl_data['Total_FTE'].sum()

            # Impact metrics
            tl_unbilled_impact = (total_tl_unbilled_fte / total_tl_fte * 100) if total_tl_fte > 0 else 0
            portfolio_unbilled_fte = prism_data[prism_data['Status'] == 'Unbilled']['Unbilled_FTE'].sum()
            tl_share_of_unbilled = (total_tl_unbilled_fte / portfolio_unbilled_fte * 100) if portfolio_unbilled_fte > 0 else 0

            # Strategic breakdown for TL unbilled
            tl_unbilled_by_category = tl_unbilled.groupby('assignstatetagdesc')['Unbilled_FTE'].sum().sort_values(ascending=False)

            narrative = self._generate_tl_impact_narrative(total_tl_unbilled_fte, tl_unbilled_impact, tl_share_of_unbilled, tl_unbilled_by_category, scope)

            return {
                'scope': scope,
                'total_tl_fte': total_tl_fte,
                'total_tl_unbilled_fte': total_tl_unbilled_fte,
                'tl_unbilled_impact_%': round(tl_unbilled_impact, 1),
                'tl_share_of_unbilled_%': round(tl_share_of_unbilled, 1),
                'tl_unbilled_by_category': tl_unbilled_by_category.to_dict(),
                'narrative_summary': narrative,
                'critical_insights': self._get_tl_critical_insights(tl_unbilled_impact, tl_share_of_unbilled)
            }

        except Exception as e:
            return {"error": f"TL unbilled impact analysis error: {str(e)}"}

    def _generate_tl_impact_narrative(self, total_tl_unbilled: float, tl_impact: float, tl_share: float,
                                    tl_unbilled_by_category: pd.Series, scope: str) -> str:
        """Generate narrative for TL unbilled impact"""
        narrative_parts = []

        narrative_parts.append(f"TL unbilled impact analysis for {scope}:")
        narrative_parts.append(f"Total TL unbilled FTE: {total_tl_unbilled:.1f}")
        narrative_parts.append(f"TL unbilled represents {tl_impact:.1f}% of total TL capacity and {tl_share:.1f}% of overall unbilled resources.")

        # Impact assessment
        if tl_impact > 15:
            narrative_parts.append("CRITICAL: High TL unbilled impact - immediate attention required!")
        elif tl_impact > 10:
            narrative_parts.append("WARNING: Significant TL unbilled impact - needs prioritization.")
        elif tl_impact > 5:
            narrative_parts.append("MODERATE: TL unbilled impact at manageable levels.")
        else:
            narrative_parts.append("GOOD: Low TL unbilled impact.")

        # Category insights
        if not tl_unbilled_by_category.empty:
            top_category = list(tl_unbilled_by_category.items())[0]
            narrative_parts.append(f"Primary TL unbilled category: '{top_category[0]}' with {top_category[1]:.1f} FTE.")

        # Strategic recommendation
        movable_tl_unbilled = sum(tl_unbilled_by_category.get(cat, 0) for cat in MOVABLE_CATEGORIES)
        if movable_tl_unbilled > 0:
            narrative_parts.append(f"Optimization opportunity: {movable_tl_unbilled:.1f} FTE of movable TL unbilled resources.")

        return " ".join(narrative_parts)

    def _get_tl_critical_insights(self, tl_impact: float, tl_share: float) -> List[str]:
        """Get critical insights for TL unbilled impact"""
        insights = []

        if tl_impact > 20:
            insights.append("TL unbilled exceeds 20% of TL capacity - critical leadership impact")
        elif tl_impact > 15:
            insights.append("High TL unbilled impact - significant leadership capacity tied up")

        if tl_share > 30:
            insights.append("TLs represent over 30% of total unbilled - disproportionate impact")
        elif tl_share > 20:
            insights.append("TLs represent significant portion of unbilled resources")

        return insights

    def get_detailed_unbilled_analysis(self, prism_data: pd.DataFrame, scope: str = "portfolio") -> Dict[str, Any]:
        """Get comprehensive unbilled analysis with strategic categorization"""
        try:
            analysis_data = prism_data[prism_data['Status'] == 'Unbilled'].copy()

            if analysis_data.empty:
                return {"error": "No unbilled data available"}

            # Strategic categorization
            analysis_data['strategic_category'] = analysis_data['assignstatetagdesc'].apply(
                lambda x: 'MOVABLE' if x in MOVABLE_CATEGORIES else 'UNMOVABLE' if x in UNMOVABLE_CATEGORIES else 'OTHER'
            )

            # Total unbilled summary
            total_unbilled_fte = analysis_data['Unbilled_FTE'].sum()
            total_unbilled_associates = len(analysis_data)

            # Strategic breakdown
            strategic_breakdown = analysis_data.groupby('strategic_category').agg({
                'Unbilled_FTE': 'sum',
                'DBO_Grade': 'count'
            }).rename(columns={'DBO_Grade': 'associate_count'}).to_dict('index')

            # Detailed category breakdown
            category_breakdown = analysis_data.groupby('assignstatetagdesc').agg({
                'Unbilled_FTE': 'sum',
                'DBO_Grade': 'count'
            }).rename(columns={'DBO_Grade': 'associate_count'}).sort_values('Unbilled_FTE', ascending=False)

            # MSA vs Non-MSA analysis
            msa_categories = [cat for cat in UNMOVABLE_CATEGORIES if 'MSA' in cat]
            non_msa_categories = [cat for cat in MOVABLE_CATEGORIES if 'Non MSA' in cat]

            msa_fte = analysis_data[analysis_data['assignstatetagdesc'].isin(msa_categories)]['Unbilled_FTE'].sum()
            non_msa_fte = analysis_data[analysis_data['assignstatetagdesc'].isin(non_msa_categories)]['Unbilled_FTE'].sum()

            return {
                'total_unbilled_fte': total_unbilled_fte,
                'total_unbilled_associates': total_unbilled_associates,
                'strategic_breakdown': strategic_breakdown,
                'category_breakdown': category_breakdown.to_dict('index'),
                'msa_vs_non_msa': {
                    'msa_buffer_fte': msa_fte,
                    'non_msa_buffer_fte': non_msa_fte,
                    'other_unbilled_fte': total_unbilled_fte - msa_fte - non_msa_fte
                },
                'scope': scope
            }

        except Exception as e:
            return {"error": f"Detailed unbilled analysis error: {str(e)}"}

class TargetAchievementAgent:
    def analyze_achievement_paths(self, metrics_df: pd.DataFrame, prism_data: pd.DataFrame,
                                forecast_data: pd.DataFrame = None, mapping_data: pd.DataFrame = None) -> Dict[str, Any]:
        try:
            if metrics_df.empty:
                return {"error": "No metrics data available"}

            # Use target from metrics_df (already merged from forecast) or default
            target_utilization = metrics_df['Target_Utilization%'].mean() if 'Target_Utilization%' in metrics_df.columns else 70.0

            current_utilization = metrics_df['overall_utilization_%'].mean()

            total_fte = prism_data['Total_FTE'].sum()
            utilization_gap_fte = ((target_utilization - current_utilization) / 100) * total_fte

            # Enhanced: Use strategic categorization for optimizable NBL
            optimizable_nbl = prism_data[
                (prism_data['Status'] == 'Unbilled') &
                (prism_data['assignstatetagdesc'].isin(MOVABLE_CATEGORIES))
            ]['Unbilled_FTE'].sum()

            # Enhanced: Get detailed unbilled analysis for narrative
            metric_agent = MetricCalculationAgent()
            unbilled_analysis = metric_agent.get_detailed_unbilled_analysis(prism_data, "target_analysis")

            projects_analysis = []
            for _, project in metrics_df.iterrows():
                project_name = project['Rev_Mapping']
                current_util = project['overall_utilization_%']

                # Get project-specific target if available, otherwise use average
                project_target = project.get('Target_Utilization%', target_utilization)
                gap_to_target = project_target - current_util

                if gap_to_target > 0:
                    improvement_potential = min(gap_to_target, 100 - current_util)
                    priority_score = (improvement_potential / gap_to_target) * (current_util / project_target)

                    projects_analysis.append({
                        'project': project_name,
                        'current_utilization': round(current_util, 1),
                        'target_utilization': round(project_target, 1),
                        'gap_to_target': round(gap_to_target, 1),
                        'improvement_potential': round(improvement_potential, 1),
                        'priority_score': round(priority_score, 2)
                    })

            projects_analysis.sort(key=lambda x: x['priority_score'], reverse=True)

            # Enhanced: Generate narrative summary
            narrative = self._generate_target_achievement_narrative(
                current_utilization, target_utilization, utilization_gap_fte,
                optimizable_nbl, unbilled_analysis
            )

            return {
                'current_utilization': round(current_utilization, 1),
                'target_utilization': round(target_utilization, 1),
                'utilization_gap_fte': round(utilization_gap_fte, 1),
                'optimizable_nbl_fte': round(optimizable_nbl, 1),
                'target_achievable': optimizable_nbl >= utilization_gap_fte,
                'top_projects': projects_analysis[:5],
                'analysis': f"Need {utilization_gap_fte:.1f} FTE reduction in NBL to achieve target",
                'data_source': 'forecast' if 'Target_Utilization%' in metrics_df.columns else 'default',
                'narrative_summary': narrative,
                'unbilled_analysis': unbilled_analysis
            }

        except Exception as e:
            return {"error": f"Target achievement analysis error: {str(e)}"}

    def _generate_target_achievement_narrative(self, current_util: float, target_util: float,
                                             gap_fte: float, optimizable_nbl: float,
                                             unbilled_analysis: Dict) -> str:
        """Generate narrative summary for target achievement analysis"""

        narrative_parts = []

        # Current status
        narrative_parts.append(f"Current utilization is at {current_util:.1f}%, against a target of {target_util:.1f}%.")

        # Gap analysis
        narrative_parts.append(f"To bridge this gap, you need to effectively convert or remove {gap_fte:.1f} FTE of unbilled resources.")

        # Opportunity assessment
        if 'strategic_breakdown' in unbilled_analysis:
            movable_fte = unbilled_analysis['strategic_breakdown'].get('MOVABLE', {}).get('Unbilled_FTE', 0)
            if movable_fte > 0:
                narrative_parts.append(f"You have {movable_fte:.1f} FTE of movable unbilled resources available for optimization.")

        # Feasibility and recommendation
        if optimizable_nbl >= gap_fte:
            narrative_parts.append(f"Primary Recommendation: Focus on converting {gap_fte:.1f} FTE from movable unbilled categories to billed work. This is sufficient to achieve your target.")
        else:
            shortfall = gap_fte - optimizable_nbl
            narrative_parts.append(f"Challenge: You only have {optimizable_nbl:.1f} FTE of readily optimizable resources, falling short by {shortfall:.1f} FTE. Additional strategies may be needed.")

        return " ".join(narrative_parts)

class ResourceOptimizationAgent:

    def optimize_nbl_removal(self, nbl_count: int, scope: List[str],
                       prism_data: pd.DataFrame, subcon_data: pd.DataFrame,
                       metric_agent: MetricCalculationAgent) -> Dict[str, Any]:
        try:
            # FIX: Handle None nbl_count properly
            if nbl_count is None:
                nbl_count = 0

            if scope and 'Rev_Mapping' in prism_data.columns:
                scope_data = prism_data[prism_data['Rev_Mapping'].isin(scope)]
            else:
                scope_data = prism_data

            print(f"Optimizing {nbl_count} NBL removal from scope: {scope}")

            # FIX: For buffer analysis queries with nbl_count=0, provide analysis without removal plan
            if nbl_count == 0:
                return self._provide_buffer_analysis_only(scope_data, subcon_data, metric_agent)

            current_metrics = metric_agent.calculate_all_metrics(scope_data, subcon_data)

            # Enhanced: Use strategic categorization
            unbilled_breakdown = scope_data[scope_data['Status'] == 'Unbilled'].groupby(
                'assignstatetagdesc'
            )['Unbilled_FTE'].sum().sort_values(ascending=False)

            # Enhanced: Get associate counts
            unbilled_associate_counts = scope_data[scope_data['Status'] == 'Unbilled'].groupby(
                'assignstatetagdesc'
            ).size().sort_values(ascending=False)

            print(f"Current unbilled breakdown: {unbilled_breakdown.to_dict()}")

            removal_plan = []
            remaining_to_remove = nbl_count

            # Only create removal plan if nbl_count > 0
            if nbl_count > 0:
                # Enhanced: Strategic removal prioritizing movable categories
                for category in MOVABLE_CATEGORIES:
                    if category in unbilled_breakdown and remaining_to_remove > 0:
                        available_fte = unbilled_breakdown[category]
                        remove_from_category = min(available_fte, remaining_to_remove)

                        if remove_from_category > 0:
                            associate_count = unbilled_associate_counts.get(category, 0)
                            removal_plan.append({
                                'category': category,
                                'fte_to_remove': remove_from_category,
                                'associates_affected': associate_count,
                                'impact': self._get_impact_level(category)
                            })
                            remaining_to_remove -= remove_from_category

            # Enhanced: Calculate new metrics with strategic removal
            total_removed = sum(item['fte_to_remove'] for item in removal_plan)
            new_billed_fte = scope_data['Billed_FTE'].sum()
            new_total_fte = scope_data['Total_FTE'].sum() - total_removed

            new_utilization = (new_billed_fte / new_total_fte * 100) if new_total_fte > 0 else 0
            current_utilization = (scope_data['Billed_FTE'].sum() / scope_data['Total_FTE'].sum() * 100) if scope_data['Total_FTE'].sum() > 0 else 0
            utilization_improvement = new_utilization - current_utilization

            # Enhanced: Generate narrative
            narrative = self._generate_optimization_narrative(
                nbl_count, total_removed, remaining_to_remove, current_utilization,
                new_utilization, utilization_improvement, removal_plan, unbilled_breakdown
            )

            return {
                'removal_plan': removal_plan,
                'total_removed': total_removed,
                'remaining_to_remove': remaining_to_remove,
                'current_utilization': round(current_utilization, 1),
                'new_utilization': round(new_utilization, 1),
                'utilization_improvement': round(utilization_improvement, 1),
                'success': total_removed >= nbl_count * 0.8 if nbl_count > 0 else True,
                'message': f"Removing {total_removed} FTE will improve utilization by {utilization_improvement:.1f}%" if nbl_count > 0 else "No removal requested",
                'narrative_summary': narrative,
                'available_movable_fte': sum(unbilled_breakdown.get(cat, 0) for cat in MOVABLE_CATEGORIES),
                'unbilled_breakdown': unbilled_breakdown.to_dict(),
                'unbilled_associate_counts': unbilled_associate_counts.to_dict()
            }

        except Exception as e:
            return {"error": f"Resource optimization error: {str(e)}"}

    def _provide_buffer_analysis_only(self, prism_data: pd.DataFrame, subcon_data: pd.DataFrame,metric_agent: MetricCalculationAgent) -> Dict[str, Any]:
        """Provide buffer analysis without removal plan when nbl_count is 0"""
        try:
            current_metrics = metric_agent.calculate_all_metrics(prism_data, subcon_data)

            # Enhanced: Use strategic categorization
            unbilled_breakdown = prism_data[prism_data['Status'] == 'Unbilled'].groupby(
                'assignstatetagdesc'
            )['Unbilled_FTE'].sum().sort_values(ascending=False)

            # Enhanced: Get associate counts
            unbilled_associate_counts = prism_data[prism_data['Status'] == 'Unbilled'].groupby(
                'assignstatetagdesc'
            ).size().sort_values(ascending=False)

            current_utilization = (prism_data['Billed_FTE'].sum() / prism_data['Total_FTE'].sum() * 100) if prism_data['Total_FTE'].sum() > 0 else 0

            # Enhanced: Generate narrative for buffer analysis
            narrative = self._generate_buffer_analysis_narrative(current_utilization, unbilled_breakdown)

            return {
                'removal_plan': [],
                'total_removed': 0,
                'remaining_to_remove': 0,
                'current_utilization': round(current_utilization, 1),
                'new_utilization': round(current_utilization, 1),
                'utilization_improvement': 0,
                'success': True,
                'message': "Buffer analysis provided - no removal requested",
                'narrative_summary': narrative,
                'available_movable_fte': sum(unbilled_breakdown.get(cat, 0) for cat in MOVABLE_CATEGORIES),
                'unbilled_breakdown': unbilled_breakdown.to_dict(),
                'unbilled_associate_counts': unbilled_associate_counts.to_dict(),
                'analysis_type': 'buffer_analysis'
            }

        except Exception as e:
            return {"error": f"Buffer analysis error: {str(e)}"}

    def _generate_buffer_analysis_narrative(self, current_util: float, unbilled_breakdown: pd.Series) -> str:
            """Generate narrative specifically for buffer analysis queries"""

            narrative_parts = []

            # Current status
            narrative_parts.append(f"Current utilization: {current_util:.1f}%.")

            # Buffer analysis
            available_movable = sum(unbilled_breakdown.get(cat, 0) for cat in MOVABLE_CATEGORIES)
            available_unmovable = sum(unbilled_breakdown.get(cat, 0) for cat in UNMOVABLE_CATEGORIES)

            narrative_parts.append(f"Available for optimization: {available_movable:.1f} FTE in movable categories.")
            narrative_parts.append(f"Contractually committed: {available_unmovable:.1f} FTE in unmovable categories.")

            # MSA vs Non-MSA specific analysis
            msa_fte = unbilled_breakdown.get('Unbilled - MSA Buffer', 0)
            non_msa_fte = unbilled_breakdown.get('Unbilled - Non MSA Buffer', 0)

            if msa_fte > 0 or non_msa_fte > 0:
                narrative_parts.append(f"MSA Buffer: {msa_fte:.1f} FTE | Non-MSA Buffer: {non_msa_fte:.1f} FTE")

            # Strategic recommendation
            if non_msa_fte > 0:
                narrative_parts.append(f"Primary optimization opportunity: Utilize the {non_msa_fte:.1f} FTE Non-MSA Buffer.")
            elif available_movable > 0:
                narrative_parts.append(f"Optimization opportunity: {available_movable:.1f} FTE available in movable categories.")
            else:
                narrative_parts.append("Limited optimization opportunities in movable categories.")

            return " ".join(narrative_parts)

    def _get_impact_level(self, category: str) -> str:
        """Get impact level for removal category"""
        impact_map = {
            "Unbilled - Non MSA Buffer": "LOW (Internal flexibility)",
            "Unbilled - KT": "MEDIUM (Knowledge transfer impact)",
            "Unbilled - Management": "HIGH (Leadership impact)"
        }
        return impact_map.get(category, "MEDIUM")

    def _generate_optimization_narrative(self, requested_removal: float, actual_removal: float,
                                       remaining: float, current_util: float, new_util: float,
                                       improvement: float, removal_plan: List[Dict],
                                       unbilled_breakdown: pd.Series) -> str:
        """Generate narrative summary for optimization analysis"""

        narrative_parts = []

        if requested_removal > 0:
            # Feasibility context for removal requests
            available_movable = sum(unbilled_breakdown.get(cat, 0) for cat in MOVABLE_CATEGORIES)
            narrative_parts.append(f"Requested removal: {requested_removal:.1f} FTE. Available movable unbilled resources: {available_movable:.1f} FTE.")

            if actual_removal < requested_removal:
                narrative_parts.append(f"Only {actual_removal:.1f} FTE can be removed from movable categories, falling short by {remaining:.1f} FTE.")
            else:
                narrative_parts.append(f"Full requested amount of {actual_removal:.1f} FTE can be removed from movable categories.")

            # Removal plan summary
            if removal_plan:
                primary_category = removal_plan[0]['category']
                narrative_parts.append(f"The removal plan prioritizes {primary_category} with {removal_plan[0]['fte_to_remove']:.1f} FTE.")

            # Outcome impact
            narrative_parts.append(f"This optimization would increase utilization from {current_util:.1f}% to {new_util:.1f}% (+{improvement:.1f}%).")
        else:
            # Analysis-only mode (for buffer analysis queries)
            available_movable = sum(unbilled_breakdown.get(cat, 0) for cat in MOVABLE_CATEGORIES)
            available_unmovable = sum(unbilled_breakdown.get(cat, 0) for cat in UNMOVABLE_CATEGORIES)

            narrative_parts.append(f"Current utilization: {current_util:.1f}%.")
            narrative_parts.append(f"Available for optimization: {available_movable:.1f} FTE in movable categories.")
            narrative_parts.append(f"Contractually committed: {available_unmovable:.1f} FTE in unmovable categories.")

            # MSA vs Non-MSA specific analysis
            msa_fte = unbilled_breakdown.get('Unbilled - MSA Buffer', 0)
            non_msa_fte = unbilled_breakdown.get('Unbilled - Non MSA Buffer', 0)

            if msa_fte > 0 or non_msa_fte > 0:
                narrative_parts.append(f"MSA Buffer: {msa_fte:.1f} FTE | Non-MSA Buffer: {non_msa_fte:.1f} FTE")

        return " ".join(narrative_parts)

class BufferAnalysisAgent:
    def __init__(self, metric_agent: MetricCalculationAgent):
        self.metric_agent = metric_agent

    def analyze_buffers(self, prism_data: pd.DataFrame, scope: str = "portfolio") -> Dict[str, Any]:
        """Specialized analysis for MSA vs Non-MSA buffers"""
        try:
            # Get detailed unbilled analysis
            unbilled_analysis = self.metric_agent.get_detailed_unbilled_analysis(prism_data, scope)

            if "error" in unbilled_analysis:
                return unbilled_analysis

            # Extract buffer-specific information
            buffer_analysis = self._extract_buffer_metrics(unbilled_analysis)

            # Generate narrative
            narrative = self._generate_buffer_narrative(buffer_analysis, unbilled_analysis)

            return {
                **buffer_analysis,
                'narrative_summary': narrative,
                'detailed_breakdown': unbilled_analysis,
                'scope': scope,
                'analysis_timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {"error": f"Buffer analysis error: {str(e)}"}

    def _extract_buffer_metrics(self, unbilled_analysis: Dict) -> Dict[str, Any]:
        """Extract MSA and Non-MSA buffer metrics - FIXED VERSION"""
        try:
            msa_fte = 0
            non_msa_fte = 0
            other_movable_fte = 0
            unmovable_fte = 0

            # Extract from category breakdown - FIXED LOGIC
            if 'category_breakdown' in unbilled_analysis:
                for category, data in unbilled_analysis['category_breakdown'].items():
                    fte = data['Unbilled_FTE']

                    # FIXED: Better category matching
                    if 'MSA Buffer' in str(category):
                        msa_fte += fte
                    elif 'Non MSA Buffer' in str(category):
                        non_msa_fte += fte
                    elif any(movable_cat in str(category) for movable_cat in MOVABLE_CATEGORIES):
                        other_movable_fte += fte
                    elif any(unmovable_cat in str(category) for unmovable_cat in UNMOVABLE_CATEGORIES):
                        unmovable_fte += fte
                    else:
                        # Default classification for uncategorized
                        if 'Buffer' in str(category):
                            non_msa_fte += fte  # Assume non-MSA if not specified
                        else:
                            other_movable_fte += fte

            # Also check strategic breakdown for totals as fallback
            if 'strategic_breakdown' in unbilled_analysis:
                strategic = unbilled_analysis['strategic_breakdown']
                movable_from_strategic = strategic.get('MOVABLE', {}).get('Unbilled_FTE', 0)
                unmovable_from_strategic = strategic.get('UNMOVABLE', {}).get('Unbilled_FTE', 0)

                # Use strategic breakdown if category breakdown seems incomplete
                if msa_fte + non_msa_fte == 0 and movable_from_strategic + unmovable_from_strategic > 0:
                    print("Using strategic breakdown as fallback for buffer analysis")
                    # Estimate MSA/Non-MSA split (typically 30% MSA, 70% Non-MSA in movable)
                    non_msa_fte = movable_from_strategic * 0.7
                    msa_fte = movable_from_strategic * 0.3
                    other_movable_fte = 0

            # Final validation
            total_calculated = msa_fte + non_msa_fte + other_movable_fte + unmovable_fte
            total_actual = unbilled_analysis.get('total_unbilled_fte', 0)

            # If there's a significant discrepancy, redistribute
            if total_actual > 0 and abs(total_calculated - total_actual) / total_actual > 0.1:
                print(f"Buffer analysis discrepancy: calculated {total_calculated}, actual {total_actual}")
                scale_factor = total_actual / total_calculated if total_calculated > 0 else 1
                msa_fte *= scale_factor
                non_msa_fte *= scale_factor
                other_movable_fte *= scale_factor
                unmovable_fte *= scale_factor

            return {
                'msa_buffer_fte': round(msa_fte, 1),
                'non_msa_buffer_fte': round(non_msa_fte, 1),
                'other_movable_fte': round(other_movable_fte, 1),
                'total_movable_fte': round(non_msa_fte + other_movable_fte, 1),
                'total_unmovable_fte': round(msa_fte + unmovable_fte, 1),
                'total_unbilled_fte': unbilled_analysis.get('total_unbilled_fte', 0),
                'total_unbilled_associates': unbilled_analysis.get('total_unbilled_associates', 0),
                'optimization_opportunity': round(non_msa_fte + other_movable_fte, 1),
                'analysis_quality': 'HIGH' if total_calculated > 0 else 'LOW'
            }

        except Exception as e:
            print(f"Error extracting buffer metrics: {e}")
            return {}

    def _generate_buffer_narrative(self, buffer_analysis: Dict, unbilled_analysis: Dict) -> str:
        """Generate narrative specifically for buffer analysis"""

        narrative_parts = []

        total_unbilled = buffer_analysis.get('total_unbilled_fte', 0)
        msa_fte = buffer_analysis.get('msa_buffer_fte', 0)
        non_msa_fte = buffer_analysis.get('non_msa_buffer_fte', 0)
        other_movable = buffer_analysis.get('other_movable_fte', 0)
        total_movable = buffer_analysis.get('total_movable_fte', 0)

        narrative_parts.append(f"Buffer analysis shows {total_unbilled:.1f} FTE total unbilled resources.")

        # MSA vs Non-MSA breakdown
        narrative_parts.append(f"MSA Buffer (contractual): {msa_fte:.1f} FTE")
        narrative_parts.append(f"Non-MSA Buffer (optimizable): {non_msa_fte:.1f} FTE")

        if other_movable > 0:
            narrative_parts.append(f"Other movable resources: {other_movable:.1f} FTE")

        # Optimization summary
        if total_movable > 0:
            narrative_parts.append(f"Total optimization opportunity: {total_movable:.1f} FTE in movable categories.")

        # Strategic recommendation
        if non_msa_fte > 0:
            narrative_parts.append(f"Primary focus: Utilize the {non_msa_fte:.1f} FTE Non-MSA Buffer for utilization improvement.")
        elif total_movable > 0:
            narrative_parts.append(f"Focus on optimizing {total_movable:.1f} FTE from other movable categories.")
        else:
            narrative_parts.append("Limited optimization opportunities in movable categories.")

        return " ".join(narrative_parts)

class UnbilledAnalysisAgent:
    def __init__(self, metric_agent: MetricCalculationAgent):
        self.metric_agent = metric_agent

    def analyze_unbilled_resources(self, prism_data: pd.DataFrame, scope: str = "portfolio") -> Dict[str, Any]:
        """Comprehensive unbilled analysis with strategic categorization"""
        try:
            detailed_analysis = self.metric_agent.get_detailed_unbilled_analysis(prism_data, scope)

            if "error" in detailed_analysis:
                return detailed_analysis

            # Generate narrative summary
            narrative = self._generate_unbilled_narrative(detailed_analysis)

            return {
                **detailed_analysis,
                'narrative_summary': narrative,
                'scope': scope,
                'analysis_timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return {"error": f"Unbilled analysis error: {str(e)}"}

    def _generate_unbilled_narrative(self, analysis: Dict) -> str:
        """Generate narrative summary for unbilled analysis"""

        narrative_parts = []

        total_fte = analysis['total_unbilled_fte']
        total_associates = analysis['total_unbilled_associates']

        narrative_parts.append(f"Unbilled analysis shows {total_fte:.1f} FTE across {total_associates} associates.")

        # Strategic breakdown narrative
        if 'strategic_breakdown' in analysis:
            strategic = analysis['strategic_breakdown']
            movable_fte = strategic.get('MOVABLE', {}).get('Unbilled_FTE', 0)
            unmovable_fte = strategic.get('UNMOVABLE', {}).get('Unbilled_FTE', 0)

            if movable_fte > 0:
                narrative_parts.append(f"Of this, {movable_fte:.1f} FTE is optimizable (movable), representing your primary lever for utilization improvement.")

            if unmovable_fte > 0:
                narrative_parts.append(f"The remaining {unmovable_fte:.1f} FTE is contractually committed or administrative (unmovable).")

        # MSA vs Non-MSA narrative
        if 'msa_vs_non_msa' in analysis:
            msa_data = analysis['msa_vs_non_msa']
            msa_fte = msa_data['msa_buffer_fte']
            non_msa_fte = msa_data['non_msa_buffer_fte']

            if msa_fte > 0:
                narrative_parts.append(f"MSA Buffer: {msa_fte:.1f} FTE (fixed contractual commitment).")

            if non_msa_fte > 0:
                narrative_parts.append(f"Non-MSA Buffer: {non_msa_fte:.1f} FTE (primary optimization opportunity).")

        # Category insights
        if 'category_breakdown' in analysis:
            categories = analysis['category_breakdown']
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1]['Unbilled_FTE'])
                narrative_parts.append(f"The largest unbilled category is '{top_category[0]}' with {top_category[1]['Unbilled_FTE']:.1f} FTE.")

        return " ".join(narrative_parts)

class EDLPerformanceAnalyzer:
    def __init__(self, metric_agent: MetricCalculationAgent):
        self.metric_agent = metric_agent
        self.tl_plus_grades = ['GM', 'SDM', 'DGM', 'TM', 'Cont', 'SD', 'AVP', 'VP', 'Dir']

    def analyze_edl_performance(self, edl_name: str, prism_data: pd.DataFrame,
                              subcon_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            prism_data_copy = prism_data.copy()

            # If no specific EDL mentioned, analyze all EDLs
            if not edl_name or edl_name.lower() == "all":
                return self._analyze_all_edls(prism_data_copy, subcon_data)

            # Single EDL analysis
            edl_mask = prism_data_copy['EDL_Name'] == edl_name
            edl_data = prism_data_copy[edl_mask].copy()

            if edl_data.empty:
                return {"error": f"No data found for EDL: {edl_name}"}

            edl_summary = self._get_edl_summary(edl_data, subcon_data)
            location_analysis = self._analyze_edl_locations(edl_data, subcon_data)
            project_analysis = self._analyze_edl_projects(edl_data, subcon_data)
            nbl_analysis = self._analyze_edl_nbl(edl_data)
            rev_mapping_analysis = self._analyze_edl_rev_mappings(edl_data, subcon_data)
            grade_analysis = self._analyze_edl_grades(edl_data)

            # NEW: Enhanced analysis components
            onsite_offshore_analysis = self._analyze_onsite_offshore(edl_data)
            associate_distribution = self._analyze_associate_distribution(edl_data)
            complete_metrics = self._get_complete_metrics_table(edl_data, subcon_data)

            # Enhanced: Generate narrative summary
            narrative = self._generate_comprehensive_edl_narrative(
                edl_summary, nbl_analysis, project_analysis,
                onsite_offshore_analysis, associate_distribution, location_analysis
            )

            return {
                'edl_name': edl_name,
                'summary': edl_summary,
                'complete_metrics_table': complete_metrics.to_dict('records'),
                'location_breakdown': location_analysis,
                'project_performance': project_analysis,
                'nbl_analysis': nbl_analysis,
                'portfolio_analysis': rev_mapping_analysis,
                'grade_analysis': grade_analysis,
                'onsite_offshore_analysis': onsite_offshore_analysis,
                'associate_distribution': associate_distribution,
                'key_insights': self._generate_edl_insights(edl_summary, nbl_analysis, project_analysis),
                'narrative_summary': narrative,
                'metrics_columns': list(complete_metrics.columns) if not complete_metrics.empty else []
            }

        except Exception as e:
            return {"error": f"EDL analysis error: {str(e)}"}

    def analyze_project_distribution(self, edl_name: str, prism_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced project distribution analysis"""
        try:
            edl_data = prism_data[prism_data['EDL_Name'] == edl_name].copy()

            if edl_data.empty:
                return {"error": f"No data found for EDL: {edl_name}"}

            # Project distribution by FTE
            project_distribution = edl_data.groupby('Project_Name').agg({
                'Total_FTE': 'sum',
                'Billed_FTE': 'sum',
                'Unbilled_FTE': 'sum',
                'DBO_Grade': 'count'
            }).rename(columns={'DBO_Grade': 'associate_count'}).sort_values('Total_FTE', ascending=False)

            # Calculate utilization for each project
            project_distribution['utilization_%'] = (project_distribution['Billed_FTE'] / project_distribution['Total_FTE'] * 100).round(1)

            # Concentration analysis
            total_fte = project_distribution['Total_FTE'].sum()
            top_project_share = (project_distribution['Total_FTE'].iloc[0] / total_fte * 100) if total_fte > 0 else 0

            # Generate narrative
            narrative = self._generate_project_distribution_narrative(edl_name, project_distribution, total_fte, top_project_share)

            return {
                'edl_name': edl_name,
                'total_projects': len(project_distribution),
                'total_fte': total_fte,
                'project_distribution': project_distribution.to_dict('index'),
                'concentration_metrics': {
                    'top_project_share_%': round(top_project_share, 1),
                    'top_project_name': project_distribution.index[0],
                    'projects_above_10pct': len([pct for pct in (project_distribution['Total_FTE'] / total_fte * 100) if pct > 10])
                },
                'narrative_summary': narrative
            }

        except Exception as e:
            return {"error": f"Project distribution analysis error: {str(e)}"}

    def analyze_location_breakdown(self, edl_name: str, prism_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced location breakdown analysis"""
        try:
            edl_data = prism_data[prism_data['EDL_Name'] == edl_name].copy()

            if edl_data.empty:
                return {"error": f"No data found for EDL: {edl_name}"}

            # Location breakdown
            location_analysis = edl_data.groupby('Utilization_Location').agg({
                'Total_FTE': 'sum',
                'Billed_FTE': 'sum',
                'Unbilled_FTE': 'sum',
                'DBO_Grade': 'count'
            }).rename(columns={'DBO_Grade': 'associate_count'}).sort_values('Total_FTE', ascending=False)

            # Calculate metrics
            location_analysis['utilization_%'] = (location_analysis['Billed_FTE'] / location_analysis['Total_FTE'] * 100).round(1)
            location_analysis['unbilled_%'] = (location_analysis['Unbilled_FTE'] / location_analysis['Total_FTE'] * 100).round(1)

            # Performance variation analysis
            util_std = location_analysis['utilization_%'].std()
            max_util = location_analysis['utilization_%'].max()
            min_util = location_analysis['utilization_%'].min()

            narrative = self._generate_location_breakdown_narrative(edl_name, location_analysis, util_std, max_util, min_util)

            return {
                'edl_name': edl_name,
                'total_locations': len(location_analysis),
                'location_breakdown': location_analysis.to_dict('index'),
                'performance_metrics': {
                    'utilization_std': round(util_std, 1),
                    'utilization_range': round(max_util - min_util, 1),
                    'best_performing_location': location_analysis['utilization_%'].idxmax(),
                    'needs_attention_location': location_analysis['utilization_%'].idxmin()
                },
                'narrative_summary': narrative
            }

        except Exception as e:
            return {"error": f"Location breakdown analysis error: {str(e)}"}

    def _generate_project_distribution_narrative(self, edl_name: str, project_distribution: pd.DataFrame, total_fte: float, top_project_share: float) -> str:
        """Generate narrative for project distribution analysis"""
        narrative_parts = []

        narrative_parts.append(f"EDL {edl_name} manages {len(project_distribution)} projects with {total_fte:.1f} total FTE.")

        # Concentration analysis
        if top_project_share > 40:
            narrative_parts.append(f"High concentration risk: {project_distribution.index[0]} represents {top_project_share:.1f}% of total FTE.")
        elif top_project_share > 25:
            narrative_parts.append(f"Moderate concentration: {project_distribution.index[0]} represents {top_project_share:.1f}% of total FTE.")
        else:
            narrative_parts.append(f"Good project diversification with {top_project_share:.1f}% in the largest project.")

        # Utilization insights
        low_util_projects = project_distribution[project_distribution['utilization_%'] < 60]
        if not low_util_projects.empty:
            narrative_parts.append(f"{len(low_util_projects)} projects have utilization below 60%.")

        # Top project performance
        top_project = project_distribution.iloc[0]
        narrative_parts.append(f"Largest project '{project_distribution.index[0]}' has {top_project['utilization_%']}% utilization with {top_project['Total_FTE']:.1f} FTE.")

        return " ".join(narrative_parts)

    def _generate_location_breakdown_narrative(self, edl_name: str, location_analysis: pd.DataFrame, util_std: float, max_util: float, min_util: float) -> str:
        """Generate narrative for location breakdown analysis"""
        narrative_parts = []

        narrative_parts.append(f"EDL {edl_name} operates across {len(location_analysis)} locations.")

        # Performance variation
        if util_std > 15:
            narrative_parts.append(f"High performance variation ({util_std:.1f}% standard deviation) across locations.")
        elif util_std > 8:
            narrative_parts.append(f"Moderate performance variation ({util_std:.1f}% standard deviation) across locations.")
        else:
            narrative_parts.append(f"Consistent performance across locations ({util_std:.1f}% standard deviation).")

        # Specific location insights
        best_loc = location_analysis['utilization_%'].idxmax()
        worst_loc = location_analysis['utilization_%'].idxmin()
        narrative_parts.append(f"Best performing: {best_loc} ({max_util}%), Needs attention: {worst_loc} ({min_util}%).")

        # Unbilled focus
        high_unbilled_locations = location_analysis[location_analysis['unbilled_%'] > 20]
        if not high_unbilled_locations.empty:
            narrative_parts.append(f"{len(high_unbilled_locations)} locations have unbilled rates above 20%.")

        return " ".join(narrative_parts)

    def _analyze_all_edls(self, prism_data: pd.DataFrame, subcon_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all EDLs and provide comparative analysis"""
        try:
            all_edls = prism_data['EDL_Name'].unique()
            print(f"Analyzing all {len(all_edls)} EDLs: {list(all_edls)}")

            all_edl_results = []
            complete_metrics_list = []

            for edl_name in all_edls:
                print(f"Analyzing EDL: {edl_name}")
                edl_mask = prism_data['EDL_Name'] == edl_name
                edl_data = prism_data[edl_mask].copy()

                if not edl_data.empty:
                    # Get basic summary for each EDL
                    edl_summary = self._get_edl_summary(edl_data, subcon_data)
                    edl_metrics = self.metric_agent.calculate_all_metrics(edl_data, subcon_data, "EDL_Name")

                    if not edl_metrics.empty:
                        metrics_row = edl_metrics.iloc[0].to_dict()
                        metrics_row['EDL_Name'] = edl_name
                        metrics_row['associate_count'] = len(edl_data)
                        metrics_row['project_count'] = edl_data['Project_Name'].nunique()
                        complete_metrics_list.append(metrics_row)

                    all_edl_results.append({
                        'edl_name': edl_name,
                        'summary': edl_summary,
                        'associate_count': len(edl_data),
                        'project_count': edl_data['Project_Name'].nunique()
                    })

            # Create comprehensive metrics table for all EDLs
            complete_metrics_df = pd.DataFrame(complete_metrics_list)

            # Generate comparative narrative
            narrative = self._generate_all_edls_narrative(all_edl_results, complete_metrics_df)

            return {
                'analysis_type': 'all_edls_comparison',
                'edl_count': len(all_edls),
                'edl_summaries': all_edl_results,
                'complete_metrics_table': complete_metrics_df.to_dict('records'),
                'comparison_metrics': self._compare_edls(all_edl_results),
                'narrative_summary': narrative,
                'metrics_columns': list(complete_metrics_df.columns) if not complete_metrics_df.empty else []
            }

        except Exception as e:
            return {"error": f"All EDLs analysis error: {str(e)}"}

    def _compare_edls(self, edl_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple EDLs and provide comparative analysis"""
        try:
            comparison_data = []

            for edl_analysis in edl_analyses:
                if 'error' not in edl_analysis and 'summary' in edl_analysis:
                    summary = edl_analysis['summary']
                    comparison_data.append({
                        'edl_name': edl_analysis['edl_name'],
                        'overall_utilization': summary.get('overall_utilization', 0),
                        'nbl_utilization': summary.get('nbl_utilization', 0),
                        'total_fte': summary.get('total_fte', 0),
                        'total_projects': summary.get('total_projects', 0),
                        'performance_tier': summary.get('performance_tier', 'UNKNOWN')
                    })

            if comparison_data:
                # Sort by utilization (descending)
                comparison_data.sort(key=lambda x: x['overall_utilization'], reverse=True)

                return {
                    'ranking': comparison_data,
                    'top_performer': comparison_data[0] if comparison_data else None,
                    'needs_attention': [edl for edl in comparison_data if edl['overall_utilization'] < 70],
                    'average_utilization': sum(edl['overall_utilization'] for edl in comparison_data) / len(comparison_data) if comparison_data else 0
                }

            return {}

        except Exception as e:
            print(f"Error comparing EDLs: {e}")
            return {}

    def _get_complete_metrics_table(self, edl_data: pd.DataFrame, subcon_data: pd.DataFrame) -> pd.DataFrame:
        """Get complete metrics table for EDL analysis"""
        try:
            # Calculate all metrics grouped by Rev_Mapping within the EDL
            metrics_df = self.metric_agent.calculate_all_metrics(edl_data, subcon_data, "Rev_Mapping")

            if metrics_df.empty:
                return pd.DataFrame()

            # Add additional metrics
            for rev_mapping in metrics_df['Rev_Mapping']:
                rev_data = edl_data[edl_data['Rev_Mapping'] == rev_mapping]

                # Add associate count
                metrics_df.loc[metrics_df['Rev_Mapping'] == rev_mapping, 'associate_count'] = len(rev_data)

                # Add project count
                metrics_df.loc[metrics_df['Rev_Mapping'] == rev_mapping, 'project_count'] = rev_data['Project_Name'].nunique()

                # Add location distribution
                location_counts = rev_data['Utilization_Location'].value_counts()
                if not location_counts.empty:
                    primary_location = location_counts.index[0]
                    metrics_df.loc[metrics_df['Rev_Mapping'] == rev_mapping, 'primary_location'] = primary_location

            # Sort by utilization (descending)
            metrics_df = metrics_df.sort_values('overall_utilization_%', ascending=False)

            return metrics_df

        except Exception as e:
            print(f"Error creating complete metrics table: {e}")
            return pd.DataFrame()

    def _analyze_onsite_offshore(self, edl_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze onsite vs offshore distribution"""
        try:
            if 'Is_Onsite' not in edl_data.columns:
                return {"error": "Onsite/Offshore data not available"}

            onsite_analysis = edl_data.groupby('Is_Onsite').agg({
                'Total_FTE': 'sum',
                'Billed_FTE': 'sum',
                'Unbilled_FTE': 'sum',
                'DBO_Grade': 'count'
            }).rename(columns={'DBO_Grade': 'associate_count'})

            # Calculate utilization for each category
            onsite_analysis['utilization_%'] = (onsite_analysis['Billed_FTE'] / onsite_analysis['Total_FTE'] * 100).round(1)

            return onsite_analysis.to_dict('index')

        except Exception as e:
            return {"error": f"Onsite/offshore analysis error: {str(e)}"}

    def _analyze_associate_distribution(self, edl_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associate distribution across various dimensions"""
        try:
            distribution = {
                'total_associates': len(edl_data),
                'by_grade': edl_data['DBO_Grade'].value_counts().to_dict(),
                'by_location': edl_data['Utilization_Location'].value_counts().to_dict(),
                'by_project': edl_data['Project_Name'].value_counts().head(10).to_dict(),
                'by_rev_mapping': edl_data['Rev_Mapping'].value_counts().to_dict()
            }

            # Calculate concentration metrics
            total_associates = distribution['total_associates']
            if total_associates > 0:
                top_project_pct = (list(distribution['by_project'].values())[0] / total_associates * 100) if distribution['by_project'] else 0
                top_location_pct = (list(distribution['by_location'].values())[0] / total_associates * 100) if distribution['by_location'] else 0

                distribution['concentration_metrics'] = {
                    'largest_project_%': round(top_project_pct, 1),
                    'largest_location_%': round(top_location_pct, 1),
                    'project_diversity': len(distribution['by_project']),
                    'location_diversity': len(distribution['by_location'])
                }

            return distribution

        except Exception as e:
            return {"error": f"Associate distribution analysis error: {str(e)}"}

    def _generate_comprehensive_edl_narrative(self, summary: Dict, nbl_analysis: Dict,
                                            project_analysis: List[Dict], onsite_offshore_analysis: Dict,
                                            associate_distribution: Dict, location_analysis: List[Dict]) -> str:
        """Generate comprehensive narrative with noteworthy insights"""

        narrative_parts = []

        # Basic performance summary
        util = summary.get('overall_utilization', 0)
        total_fte = summary.get('total_fte', 0)
        total_associates = associate_distribution.get('total_associates', 0)
        total_projects = summary.get('total_projects', 0)

        narrative_parts.append(
            f"EDL manages {total_associates} associates across {total_projects} projects with {total_fte:.1f} total FTE. "
            f"Current utilization: {util:.1f}%."
        )

        # Performance assessment
        if util >= 75:
            narrative_parts.append("Performance is excellent, exceeding utilization targets.")
        elif util >= 70:
            narrative_parts.append("Performance is good, meeting utilization targets.")
        elif util >= 60:
            narrative_parts.append("Performance needs improvement to reach target levels.")
        else:
            narrative_parts.append("Performance requires immediate attention.")

        # Associate distribution insights
        if 'concentration_metrics' in associate_distribution:
            conc_metrics = associate_distribution['concentration_metrics']
            if conc_metrics['largest_project_%'] > 40:
                narrative_parts.append(
                    f"Note: High project concentration - largest project represents {conc_metrics['largest_project_%']}% of associates."
                )
            if conc_metrics['largest_location_%'] > 60:
                narrative_parts.append(
                    f"Geographic concentration - {conc_metrics['largest_location_%']}% of associates in primary location."
                )

        # Onsite/Offshore insights
        if 'Onsite' in onsite_offshore_analysis and 'Offshore' in onsite_offshore_analysis:
            onsite_util = onsite_offshore_analysis.get('Onsite', {}).get('utilization_%', 0)
            offshore_util = onsite_offshore_analysis.get('Offshore', {}).get('utilization_%', 0)

            if abs(onsite_util - offshore_util) > 15:
                narrative_parts.append(
                    f"Significant utilization gap: Onsite {onsite_util}% vs Offshore {offshore_util}%."
                )

        # NBL optimization opportunities
        movable_fte = nbl_analysis.get('movable_unbilled_fte', 0)
        if movable_fte > 5:
            narrative_parts.append(
                f"Optimization opportunity: {movable_fte:.1f} FTE available in movable unbilled categories."
            )

        # Location performance variation
        if len(location_analysis) > 1:
            location_utils = [loc['utilization'] for loc in location_analysis]
            util_range = max(location_utils) - min(location_utils)
            if util_range > 20:
                narrative_parts.append(
                    f"High location performance variation ({util_range:.1f}% spread) - review location strategy."
                )

        return " ".join(narrative_parts)

    def _generate_all_edls_narrative(self, all_edl_results: List[Dict], complete_metrics_df: pd.DataFrame) -> str:
        """Generate narrative for all EDLs analysis"""

        narrative_parts = []

        if not all_edl_results:
            return "No EDL data available for analysis."

        narrative_parts.append(f"Comparative analysis of {len(all_edl_results)} EDLs:")

        # Overall statistics
        total_associates = sum(edl.get('associate_count', 0) for edl in all_edl_results)
        avg_utilization = sum(edl.get('summary', {}).get('overall_utilization', 0) for edl in all_edl_results) / len(all_edl_results)

        narrative_parts.append(
            f"Total {total_associates} associates with average utilization of {avg_utilization:.1f}%."
        )

        # Top performers
        sorted_edls = sorted(all_edl_results, key=lambda x: x.get('summary', {}).get('overall_utilization', 0), reverse=True)
        if sorted_edls:
            top_edl = sorted_edls[0]
            narrative_parts.append(
                f"Top performer: {top_edl['edl_name']} ({top_edl.get('summary', {}).get('overall_utilization', 0):.1f}% utilization)."
            )

        # Areas needing attention
        low_performers = [edl for edl in all_edl_results if edl.get('summary', {}).get('overall_utilization', 0) < 65]
        if low_performers:
            narrative_parts.append(
                f"{len(low_performers)} EDLs below 65% utilization requiring attention."
            )

        # Size distribution insights
        edl_sizes = [edl.get('associate_count', 0) for edl in all_edl_results]
        if edl_sizes:
            avg_size = sum(edl_sizes) / len(edl_sizes)
            largest_edl = max(edl_sizes)
            smallest_edl = min(edl_sizes)

            narrative_parts.append(
                f"EDL sizes range from {smallest_edl} to {largest_edl} associates (average: {avg_size:.0f})."
            )

        return " ".join(narrative_parts)

    def _generate_edl_narrative(self, summary: Dict, nbl_analysis: Dict, project_analysis: List[Dict]) -> str:
        """Generate narrative summary for EDL analysis"""
        narrative_parts = []

        util = summary.get('overall_utilization', 0)
        total_fte = summary.get('total_fte', 0)
        total_nbl = nbl_analysis.get('total_nbl_fte', 0)

        narrative_parts.append(f"EDL performance summary: {util:.1f}% utilization with {total_fte:.1f} total FTE.")

        # Utilization status
        if util >= 75:
            narrative_parts.append("Performance is excellent, exceeding utilization targets.")
        elif util >= 70:
            narrative_parts.append("Performance is good, meeting utilization targets.")
        elif util >= 60:
            narrative_parts.append("Performance needs improvement to reach target levels.")
        else:
            narrative_parts.append("Performance requires immediate attention.")

        # NBL analysis
        if total_nbl > 0:
            movable_fte = nbl_analysis.get('movable_unbilled_fte', 0)
            if movable_fte > 0:
                narrative_parts.append(f"Optimization opportunity: {movable_fte:.1f} FTE of movable unbilled resources available.")

        # Project concentration
        if project_analysis:
            top_project = project_analysis[0]
            if top_project['total_fte'] > total_fte * 0.3:
                narrative_parts.append(f"Note: {top_project['project_name']} represents {top_project['total_fte']/total_fte*100:.1f}% of total FTE.")

        return " ".join(narrative_parts)

    def _get_edl_summary(self, edl_data: pd.DataFrame, subcon_data: pd.DataFrame) -> Dict[str, Any]:
        edl_metrics = self.metric_agent.calculate_all_metrics(edl_data, subcon_data, "EDL_Name")

        if edl_metrics.empty:
            return {}

        summary = edl_metrics.iloc[0]

        return {
            'total_fte': edl_data['Total_FTE'].sum(),
            'billed_fte': edl_data['Billed_FTE'].sum(),
            'unbilled_fte': edl_data['Unbilled_FTE'].sum(),
            'overall_utilization': round(summary['overall_utilization_%'], 1),
            'nbl_utilization': round(summary['nbl_utilization_%'], 1),
            'nbl_tl_utilization': round(summary['nbl_tl_utilization_%'], 1),
            'total_projects': edl_data['Project_Name'].nunique(),
            'total_locations': edl_data['Utilization_Location'].nunique(),
            'total_rev_mappings': edl_data['Rev_Mapping'].nunique(),
            'performance_tier': self._get_performance_tier(summary['overall_utilization_%'])
        }

    def _analyze_edl_locations(self, edl_data: pd.DataFrame, subcon_data: pd.DataFrame) -> List[Dict[str, Any]]:
        locations_analysis = []

        for location in edl_data['Utilization_Location'].unique():
            location_mask = edl_data['Utilization_Location'] == location
            location_data = edl_data[location_mask].copy()

            location_metrics = self.metric_agent.calculate_all_metrics(location_data, subcon_data, "Utilization_Location")

            if not location_metrics.empty:
                loc_row = location_metrics.iloc[0]

                nbl_breakdown = location_data[location_data['Status'] == 'Unbilled'].groupby(
                    'assignstatetagdesc'
                )['Unbilled_FTE'].sum().sort_values(ascending=False)

                locations_analysis.append({
                    'location': location,
                    'utilization': round(loc_row['overall_utilization_%'], 1),
                    'nbl_utilization': round(loc_row['nbl_utilization_%'], 1),
                    'total_fte': location_data['Total_FTE'].sum(),
                    'billed_fte': location_data['Billed_FTE'].sum(),
                    'top_nbl_categories': nbl_breakdown.head(3).to_dict(),
                    'project_count': location_data['Project_Name'].nunique()
                })

        return sorted(locations_analysis, key=lambda x: x['utilization'])

    def _analyze_edl_projects(self, edl_data: pd.DataFrame, subcon_data: pd.DataFrame) -> List[Dict[str, Any]]:
        projects_analysis = []

        top_projects = edl_data.groupby('Project_Name').agg({
            'Total_FTE': 'sum',
            'Billed_FTE': 'sum',
            'Unbilled_FTE': 'sum'
        }).nlargest(10, 'Total_FTE').reset_index()

        for _, project in top_projects.iterrows():
            project_mask = edl_data['Project_Name'] == project['Project_Name']
            project_data = edl_data[project_mask].copy()

            project_metrics = self.metric_agent.calculate_all_metrics(project_data, subcon_data, "Project_Name")

            if not project_metrics.empty:
                proj_row = project_metrics.iloc[0]

                projects_analysis.append({
                    'project_name': project['Project_Name'],
                    'rev_mapping': project_data['Rev_Mapping'].iloc[0] if 'Rev_Mapping' in project_data.columns else 'Unknown',
                    'utilization': round(proj_row['overall_utilization_%'], 1),
                    'nbl_utilization': round(proj_row['nbl_utilization_%'], 1),
                    'total_fte': project['Total_FTE'],
                    'billed_fte': project['Billed_FTE'],
                    'unbilled_fte': project['Unbilled_FTE'],
                    'primary_location': project_data['Utilization_Location'].mode().iloc[0] if not project_data['Utilization_Location'].empty else 'Unknown',
                    'performance_tier': self._get_performance_tier(proj_row['overall_utilization_%'])
                })

        return sorted(projects_analysis, key=lambda x: x['utilization'])

    def _analyze_edl_nbl(self, edl_data: pd.DataFrame) -> Dict[str, Any]:
        unbilled_mask = edl_data['Status'] == 'Unbilled'
        unbilled_data = edl_data[unbilled_mask].copy()

        nbl_by_category = unbilled_data.groupby('assignstatetagdesc')['Unbilled_FTE'].sum().sort_values(ascending=False)
        nbl_by_grade = unbilled_data.groupby('DBO_Grade')['Unbilled_FTE'].sum().sort_values(ascending=False)
        nbl_by_location = unbilled_data.groupby('Utilization_Location')['Unbilled_FTE'].sum().sort_values(ascending=False)

        tl_nbl = unbilled_data[unbilled_data['DBO_Grade'].isin(self.tl_plus_grades)]['Unbilled_FTE'].sum()

        # Enhanced: Calculate strategic unbilled breakdown
        movable_fte = sum(nbl_by_category.get(cat, 0) for cat in MOVABLE_CATEGORIES)
        unmovable_fte = sum(nbl_by_category.get(cat, 0) for cat in UNMOVABLE_CATEGORIES)

        return {
            'total_nbl_fte': unbilled_data['Unbilled_FTE'].sum(),
            'nbl_by_category': nbl_by_category.to_dict(),
            'nbl_by_grade': nbl_by_grade.head(5).to_dict(),
            'nbl_by_location': nbl_by_location.head(5).to_dict(),
            'tl_plus_nbl': tl_nbl,
            'tl_plus_nbl_percentage': (tl_nbl / unbilled_data['Unbilled_FTE'].sum() * 100) if unbilled_data['Unbilled_FTE'].sum() > 0 else 0,
            'optimization_priority': self._get_nbl_optimization_priority(nbl_by_category),
            'movable_unbilled_fte': movable_fte,
            'unmovable_unbilled_fte': unmovable_fte
        }

    def _analyze_edl_rev_mappings(self, edl_data: pd.DataFrame, subcon_data: pd.DataFrame) -> List[Dict[str, Any]]:
        rev_mapping_analysis = []

        for rev_mapping in edl_data['Rev_Mapping'].unique():
            rev_mask = edl_data['Rev_Mapping'] == rev_mapping
            rev_data = edl_data[rev_mask].copy()

            rev_metrics = self.metric_agent.calculate_all_metrics(rev_data, subcon_data, "Rev_Mapping")

            if not rev_metrics.empty:
                rev_row = rev_metrics.iloc[0]

                rev_mapping_analysis.append({
                    'rev_mapping': rev_mapping,
                    'utilization': round(rev_row['overall_utilization_%'], 1),
                    'nbl_utilization': round(rev_row['nbl_utilization_%'], 1),
                    'total_fte': rev_data['Total_FTE'].sum(),
                    'project_count': rev_data['Project_Name'].nunique(),
                    'location_count': rev_data['Utilization_Location'].nunique(),
                    'primary_location': rev_data['Utilization_Location'].mode().iloc[0] if not rev_data['Utilization_Location'].empty else 'Unknown',
                    'performance_tier': self._get_performance_tier(rev_row['overall_utilization_%'])
                })

        return sorted(rev_mapping_analysis, key=lambda x: x['total_fte'], reverse=True)

    def _analyze_edl_grades(self, edl_data: pd.DataFrame) -> List[Dict[str, Any]]:
        grade_analysis = []

        for grade in edl_data['DBO_Grade'].unique():
            grade_mask = edl_data['DBO_Grade'] == grade
            grade_data = edl_data[grade_mask].copy()

            total_fte = grade_data['Total_FTE'].sum()
            billed_fte = grade_data['Billed_FTE'].sum()
            unbilled_fte = grade_data['Unbilled_FTE'].sum()

            utilization = (billed_fte / total_fte * 100) if total_fte > 0 else 0

            grade_analysis.append({
                'grade': grade,
                'utilization': round(utilization, 1),
                'total_fte': total_fte,
                'billed_fte': billed_fte,
                'unbilled_fte': unbilled_fte,
                'unbilled_percentage': (unbilled_fte / total_fte * 100) if total_fte > 0 else 0
            })

        return sorted(grade_analysis, key=lambda x: x['total_fte'], reverse=True)

    def _get_nbl_optimization_priority(self, nbl_by_category: pd.Series) -> List[Dict[str, Any]]:
        priorities = []

        priority_map = {
            'Unbilled - Non MSA Buffer': {'priority': 1, 'impact': 'HIGH', 'risk': 'LOW'},
            'Unbilled - KT': {'priority': 2, 'impact': 'MEDIUM', 'risk': 'MEDIUM'},
            'Unbilled - Management': {'priority': 3, 'impact': 'HIGH', 'risk': 'HIGH'},
            'Unbilled - MSA Buffer': {'priority': 4, 'impact': 'LOW', 'risk': 'HIGH'},
            'One % alloc for T&E/Prj access': {'priority': 5, 'impact': 'LOW', 'risk': 'LOW'},
            'Allocated - Awaiting billing': {'priority': 6, 'impact': 'LOW', 'risk': 'LOW'}
        }

        for category, fte in nbl_by_category.items():
            if category in priority_map:
                priorities.append({
                    'category': category,
                    'fte': round(fte, 1),
                    'priority': priority_map[category]['priority'],
                    'impact': priority_map[category]['impact'],
                    'risk': priority_map[category]['risk']
                })

        return sorted(priorities, key=lambda x: x['priority'])

    def _generate_edl_insights(self, summary: Dict, nbl_analysis: Dict, project_analysis: List[Dict]) -> List[str]:
        insights = []

        if summary['overall_utilization'] < 60:
            insights.append(f"Critical: Utilization ({summary['overall_utilization']}%) significantly below 70% target")
        elif summary['overall_utilization'] < 70:
            insights.append(f"Warning: Utilization ({summary['overall_utilization']}%) below target, needs improvement")

        if nbl_analysis['total_nbl_fte'] > 50:
            insights.append(f"High NBL: {nbl_analysis['total_nbl_fte']:.1f} FTE unbilled - major optimization opportunity")

        if len(project_analysis) > 0:
            top_project = project_analysis[0]
            if top_project['total_fte'] > summary['total_fte'] * 0.3:
                insights.append(f"Concentration risk: {top_project['project_name']} represents {top_project['total_fte']/summary['total_fte']*100:.1f}% of total FTE")

        if any(grade['unbilled_percentage'] > 30 for grade in nbl_analysis.get('grade_breakdown', [])):
            insights.append("High unbilled percentage in certain grades - review allocation strategy")

        return insights

    def _get_performance_tier(self, utilization: float) -> str:
        if utilization >= 75:
            return "EXCELLENT"
        elif utilization >= 70:
            return "GOOD"
        elif utilization >= 60:
            return "AVERAGE"
        elif utilization >= 50:
            return "POOR"
        else:
            return "CRITICAL"

class DiagnosticAgent:
    def generate_recommendations(self, metrics_df: pd.DataFrame, prism_data: pd.DataFrame,
                               analysis_type: str = "unbilled_breakdown") -> Dict[str, Any]:
        try:
            recommendations = []
            insights = []

            avg_utilization = metrics_df['overall_utilization_%'].mean()
            if avg_utilization < 70:
                insights.append(f"Portfolio utilization ({avg_utilization:.1f}%) below target 70%")
                recommendations.append("Focus on converting movable unbilled resources (Non-MSA Buffer, KT) to billed work")

            # Enhanced: Use strategic categorization for high NBL analysis
            high_nbl_projects = metrics_df[metrics_df['nbl_utilization_%'] > 25]
            if not high_nbl_projects.empty:
                insights.append(f"{len(high_nbl_projects)} projects have NBL > 25%")
                for _, project in high_nbl_projects.head(3).iterrows():
                    recommendations.append(
                        f"Optimize movable NBL in {project['Rev_Mapping']} (current: {project['nbl_utilization_%']:.1f}%)"
                    )

            # Enhanced: Analyze movable vs unmovable unbilled
            metric_agent = MetricCalculationAgent()
            unbilled_analysis = metric_agent.get_detailed_unbilled_analysis(prism_data)

            if 'strategic_breakdown' in unbilled_analysis:
                movable_fte = unbilled_analysis['strategic_breakdown'].get('MOVABLE', {}).get('Unbilled_FTE', 0)
                if movable_fte > 0:
                    insights.append(f"{movable_fte:.1f} FTE available in movable unbilled categories for optimization")
                    recommendations.append(f"Prioritize movable unbilled reduction to improve utilization")

            if 'Utilization_Location' in prism_data.columns:
                location_metrics = prism_data.groupby('Utilization_Location').agg({
                    'Billed_FTE': 'sum',
                    'Total_FTE': 'sum'
                }).reset_index()
                location_metrics['utilization_%'] = (location_metrics['Billed_FTE'] / location_metrics['Total_FTE'] * 100)
                poor_locations = location_metrics[location_metrics['utilization_%'] < 60]
                if not poor_locations.empty:
                    worst_location = poor_locations.nsmallest(1, 'utilization_%').iloc[0]
                    insights.append(f"{worst_location['Utilization_Location']} has lowest utilization ({worst_location['utilization_%']:.1f}%)")

            # Generate narrative summary
            narrative = self._generate_diagnostic_narrative(insights, recommendations, avg_utilization, len(metrics_df))

            return {
                'insights': insights,
                'recommendations': recommendations,
                'projects_analyzed': len(metrics_df),
                'average_utilization': round(avg_utilization, 1),
                'narrative_summary': narrative,
                'unbilled_analysis': unbilled_analysis
            }

        except Exception as e:
            return {"error": f"Diagnostic analysis error: {str(e)}"}

    def analyze_nbl_threshold(self, prism_data: pd.DataFrame, subcon_data: pd.DataFrame,
                            threshold: float = 25, group_by: str = "Rev_Mapping") -> Dict[str, Any]:
        """Analyze projects with NBL above specified threshold"""
        try:
            # Calculate metrics
            metric_agent = MetricCalculationAgent()
            metrics_df = metric_agent.calculate_all_metrics(prism_data, subcon_data, group_by)

            if metrics_df.empty:
                return {"error": "No metrics data available"}

            # Filter projects above threshold
            high_nbl_projects = metrics_df[metrics_df['nbl_utilization_%'] > threshold].copy()

            if high_nbl_projects.empty:
                return {
                    "message": f"No projects found with NBL > {threshold}%",
                    "threshold": threshold,
                    "total_projects_analyzed": len(metrics_df)
                }

            # Enhanced analysis
            high_nbl_projects = high_nbl_projects.sort_values('nbl_utilization_%', ascending=False)

            # Calculate total impact
            total_high_nbl_fte = high_nbl_projects['total_unbilled_fte'].sum()
            total_portfolio_fte = metrics_df['Total_FTE'].sum()
            high_nbl_share = (total_high_nbl_fte / total_portfolio_fte * 100) if total_portfolio_fte > 0 else 0

            narrative = self._generate_nbl_threshold_narrative(high_nbl_projects, threshold, total_high_nbl_fte, high_nbl_share)

            return {
                'threshold': threshold,
                'projects_above_threshold': len(high_nbl_projects),
                'total_high_nbl_fte': total_high_nbl_fte,
                'high_nbl_share_%': round(high_nbl_share, 1),
                'high_nbl_projects': high_nbl_projects.to_dict('records'),
                'narrative_summary': narrative,
                'recommendations': self._get_nbl_threshold_recommendations(high_nbl_projects, threshold)
            }

        except Exception as e:
            return {"error": f"NBL threshold analysis error: {str(e)}"}

    def _generate_nbl_threshold_narrative(self, high_nbl_projects: pd.DataFrame, threshold: float,
                                        total_high_nbl_fte: float, high_nbl_share: float) -> str:
        """Generate narrative for NBL threshold analysis"""
        narrative_parts = []

        narrative_parts.append(f"Found {len(high_nbl_projects)} projects with NBL > {threshold}%.")
        narrative_parts.append(f"Total high-NBL FTE: {total_high_nbl_fte:.1f} ({high_nbl_share:.1f}% of portfolio FTE).")

        if len(high_nbl_projects) > 0:
            top_project = high_nbl_projects.iloc[0]
            narrative_parts.append(f"Highest NBL project: {top_project.get('Rev_Mapping', 'Unknown')} with {top_project['nbl_utilization_%']:.1f}% NBL.")

        # Risk assessment
        if high_nbl_share > 20:
            narrative_parts.append(" HIGH RISK: Over 20% of portfolio FTE in high-NBL projects.")
        elif high_nbl_share > 10:
            narrative_parts.append(" MODERATE RISK: Significant portion of FTE in high-NBL projects.")

        return " ".join(narrative_parts)

    def _get_nbl_threshold_recommendations(self, high_nbl_projects: pd.DataFrame, threshold: float) -> List[str]:
        """Get recommendations for high NBL projects"""
        recommendations = []

        if len(high_nbl_projects) > 5:
            recommendations.append(f"Prioritize top 3-5 projects from {len(high_nbl_projects)} high-NBL projects")

        # Focus on projects with highest absolute NBL FTE
        if not high_nbl_projects.empty:
            max_nbl_project = high_nbl_projects.nlargest(1, 'total_unbilled_fte').iloc[0]
            recommendations.append(f"Immediate focus: {max_nbl_project.get('Rev_Mapping', 'Unknown')} with {max_nbl_project['total_unbilled_fte']:.1f} FTE unbilled")

        recommendations.append(f"Review unbilled categories and convert movable resources to billed work")
        recommendations.append("Consider resource reallocation from high-NBL to low-NBL projects")

        return recommendations

    def _generate_diagnostic_narrative(self, insights: List[str], recommendations: List[str],
                                     avg_util: float, project_count: int) -> str:
        """Generate narrative summary for diagnostic analysis"""

        narrative_parts = []

        narrative_parts.append(f"Diagnostic analysis of {project_count} projects shows an average utilization of {avg_util:.1f}%.")

        # Key insight highlight
        if insights:
            primary_insight = insights[0]
            narrative_parts.append(f"Key finding: {primary_insight}")

        # Primary recommendation
        if recommendations:
            primary_recommendation = recommendations[0]
            narrative_parts.append(f"Primary recommendation: {primary_recommendation}")

        return " ".join(narrative_parts)

class DataRetrievalAgent:
    def __init__(self, prism_data_path: str, subcon_data_path: str, mapping_data_path: str, forecast_data_path: str):
        try:
            self.prism_data = pd.read_csv(prism_data_path)
            print(f"Loaded PRISM data: {len(self.prism_data)} rows")

            # Ensure numeric columns in PRISM data
            numeric_columns = ['Billed_FTE', 'Total_FTE', 'Unbilled_FTE', 'Billed_Hours', 'Available_Hours', 'Location_Available_Hours']
            for col in numeric_columns:
                if col in self.prism_data.columns:
                    self.prism_data[col] = pd.to_numeric(self.prism_data[col], errors='coerce').fillna(0)

        except Exception as e:
            print(f"Error loading PRISM data: {e}")
            self.prism_data = pd.DataFrame()

        try:
            self.subcon_data = pd.read_csv(subcon_data_path)
            print(f"Loaded Subcon data: {len(self.subcon_data)} rows")

            # Ensure numeric columns in subcon data
            if 'subcon' in self.subcon_data.columns:
                self.subcon_data['subcon'] = pd.to_numeric(self.subcon_data['subcon'], errors='coerce').fillna(0)

        except Exception as e:
            print(f"Error loading Subcon data: {e}")
            self.subcon_data = pd.DataFrame()

        try:
            self.mapping_data = pd.read_csv(mapping_data_path)
            print(f"Loaded Mapping data: {len(self.mapping_data)} rows")
        except Exception as e:
            print(f"Error loading Mapping data: {e}")
            self.mapping_data = pd.DataFrame()

        try:
            self.forecast_data = pd.read_csv(forecast_data_path)
            print(f"Loaded Forecast data: {len(self.forecast_data)} rows")

            # Clean and convert forecast percentage columns with robust handling
            self._clean_forecast_data()

        except Exception as e:
            print(f"Error loading Forecast data: {e}")
            self.forecast_data = pd.DataFrame()

    def _clean_forecast_data(self):
        """Clean forecast data with robust type conversion"""
        try:
            print("Cleaning forecast data types...")

            # Handle Target_Utilization%
            if 'Target_Utilization%' in self.forecast_data.columns:
                print(f"Before cleaning - Target_Utilization% sample: {self.forecast_data['Target_Utilization%'].head(3).tolist()}")

                # Remove percentage signs, spaces, and convert to numeric
                self.forecast_data['Target_Utilization%'] = (
                    self.forecast_data['Target_Utilization%']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '', regex=False)
                )

                # Convert to numeric, coerce errors to NaN
                self.forecast_data['Target_Utilization%'] = pd.to_numeric(
                    self.forecast_data['Target_Utilization%'],
                    errors='coerce'
                )

                # Fill NaN with default value
                nan_count = self.forecast_data['Target_Utilization%'].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} NaN values in Target_Utilization%, filling with 70.0")
                    self.forecast_data['Target_Utilization%'] = self.forecast_data['Target_Utilization%'].fillna(70.0)

                print(f"After cleaning - Target_Utilization% dtype: {self.forecast_data['Target_Utilization%'].dtype}")
                print(f"After cleaning - Target_Utilization% range: {self.forecast_data['Target_Utilization%'].min():.1f}% to {self.forecast_data['Target_Utilization%'].max():.1f}%")

            # Handle Calculated_Forecast_Utilization% if present
            if 'Calculated_Forecast_Utilization%' in self.forecast_data.columns:
                print(f"Before cleaning - Calculated_Forecast_Utilization% sample: {self.forecast_data['Calculated_Forecast_Utilization%'].head(3).tolist()}")

                self.forecast_data['Calculated_Forecast_Utilization%'] = (
                    self.forecast_data['Calculated_Forecast_Utilization%']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '', regex=False)
                )

                self.forecast_data['Calculated_Forecast_Utilization%'] = pd.to_numeric(
                    self.forecast_data['Calculated_Forecast_Utilization%'],
                    errors='coerce'
                )

                nan_count = self.forecast_data['Calculated_Forecast_Utilization%'].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} NaN values in Calculated_Forecast_Utilization%, filling with mean")
                    mean_val = self.forecast_data['Calculated_Forecast_Utilization%'].mean()
                    self.forecast_data['Calculated_Forecast_Utilization%'] = self.forecast_data['Calculated_Forecast_Utilization%'].fillna(mean_val)

                print(f"After cleaning - Calculated_Forecast_Utilization% dtype: {self.forecast_data['Calculated_Forecast_Utilization%'].dtype}")

            # Handle other numeric columns that might be needed
            numeric_forecast_cols = ['Calculated_Billed_Hours', 'Calculated_Billed_FTE', 'Total_FTE', 'Target_Billed_FTE', 'Target_Total_FTE']
            for col in numeric_forecast_cols:
                if col in self.forecast_data.columns:
                    self.forecast_data[col] = pd.to_numeric(self.forecast_data[col], errors='coerce').fillna(0)

            print("Forecast data cleaning completed successfully")

            if 'EDL_Name' not in self.forecast_data.columns:
                print("EDL_Name not found in forecast data, attempting to merge from mapping...")
                if hasattr(self, 'mapping_data') and not self.mapping_data.empty:
                    if 'Project_Name' in self.forecast_data.columns and 'Project_Name' in self.mapping_data.columns:
                        # Merge EDL_Name from mapping data
                        before_merge = len(self.forecast_data)
                        self.forecast_data = self.forecast_data.merge(
                            self.mapping_data[['Project_Name', 'EDL_Name']].drop_duplicates(),
                            on='Project_Name',
                            how='left'
                        )
                        after_merge = len(self.forecast_data)
                        print(f"Merged EDL_Name from mapping: {before_merge} -> {after_merge} rows")
                        print(f"EDL_Name coverage: {self.forecast_data['EDL_Name'].notna().sum()}/{len(self.forecast_data)}")

            print("Forecast data cleaning completed successfully")

        except Exception as e:
            print(f"Error during forecast data cleaning: {e}")

    def get_filtered_data(self, parsed_result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get filtered data based on parsed query entities
        Returns filtered prism_data and subcon_data
        """
        try:
            entities = parsed_result.get("entities", {})

            # Start with copies to avoid SettingWithCopyWarning
            prism_filtered = self.prism_data.copy()
            subcon_filtered = self.subcon_data.copy()

            if entities.get("rev_mappings"):
                rev_mappings = entities["rev_mappings"]
                print(f"Filtering to Rev_Mappings: {rev_mappings}")

                # Use .loc for boolean indexing to avoid warnings
                prism_mask = prism_filtered['Rev_Mapping'].isin(rev_mappings)
                subcon_mask = subcon_filtered['Rev_Mapping'].isin(rev_mappings)

                prism_filtered = prism_filtered[prism_mask].copy()
                subcon_filtered = subcon_filtered[subcon_mask].copy()

            if entities.get("edl_names"):
                edl_names = entities["edl_names"]
                print(f"Filtering to EDLs: {edl_names}")

                # Use .loc for boolean indexing
                prism_mask = prism_filtered['EDL_Name'].isin(edl_names)
                prism_filtered = prism_filtered[prism_mask].copy()

            print(f"Final filtered data: {len(prism_filtered)} PRISM rows, {len(subcon_filtered)} Subcon rows")
            return prism_filtered, subcon_filtered

        except Exception as e:
            print(f"Data filtering error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()

    def get_forecast_targets_by_group(self, group_by: str) -> pd.DataFrame:
        """
        Get forecast targets aggregated by the specified grouping level
        Returns DataFrame with group_by column and Target_Utilization% column
        """
        try:
            if self.forecast_data.empty or 'Target_Utilization%' not in self.forecast_data.columns:
                print("No forecast data available for target utilization")
                return pd.DataFrame()

            # Create mapping from project to various grouping levels
            project_mapping = self.mapping_data[['Project_Name', 'Rev_Mapping', 'EDL_Name']].drop_duplicates()

            # Merge forecast with mapping to get all grouping columns
            forecast_with_mapping = self.forecast_data.merge(
                project_mapping,
                on='Project_Name',
                how='left'
            )

            # Handle case where EDL_Name might be missing after merge
            if 'EDL_Name' not in forecast_with_mapping.columns:
                print("Warning: EDL_Name not available in merged forecast data")
                # For EDL grouping, we need to use a different approach
                if group_by == "EDL_Name":
                    # Get EDL names from prism data and assign average target
                    if not self.prism_data.empty and 'EDL_Name' in self.prism_data.columns:
                        edl_list = self.prism_data['EDL_Name'].unique()
                        overall_avg = forecast_with_mapping['Target_Utilization%'].mean()
                        targets = pd.DataFrame({'EDL_Name': edl_list, 'Target_Utilization%': [overall_avg] * len(edl_list)})
                        print(f"Using overall average target {overall_avg:.1f}% for {len(edl_list)} EDLs")
                        return targets

            # Aggregate targets based on the requested grouping level
            if group_by == "Project_Name" and 'Project_Name' in forecast_with_mapping.columns:
                # Direct project-level targets
                targets = forecast_with_mapping[['Project_Name', 'Target_Utilization%']].dropna()
                print(f"Using project-level targets for {len(targets)} projects")

            elif group_by == "Rev_Mapping" and 'Rev_Mapping' in forecast_with_mapping.columns:
                # Average targets for all projects in each Rev_Mapping
                targets = (
                    forecast_with_mapping
                    .groupby('Rev_Mapping')['Target_Utilization%']
                    .mean()
                    .reset_index()
                )
                print(f"Using Rev_Mapping average targets for {len(targets)} Rev_Mappings")

            elif group_by == "EDL_Name" and 'EDL_Name' in forecast_with_mapping.columns:
                # Average targets for all projects in each EDL
                targets = (
                    forecast_with_mapping
                    .groupby('EDL_Name')['Target_Utilization%']
                    .mean()
                    .reset_index()
                )
                print(f"Using EDL average targets for {len(targets)} EDLs")

            else:
                # For other groupings, use overall average
                overall_avg = forecast_with_mapping['Target_Utilization%'].mean()
                targets = pd.DataFrame({group_by: ['overall'], 'Target_Utilization%': [overall_avg]})
                print(f"Using overall average target: {overall_avg:.1f}% for grouping: {group_by}")

            return targets

        except Exception as e:
            print(f"Error getting forecast targets by group: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

class UtilizationAnalyticsEngine:
    def __init__(self, api_key: str, data_paths: Dict[str, str]):
        self.llm_parser = LLMFirstQueryParsingAgent(api_key, data_paths['mapping'])
        self.data_agent = DataRetrievalAgent(
            data_paths['prism'],
            data_paths['subcon'],
            data_paths['mapping'],
            data_paths['forecast']
        )
        self.metric_agent = MetricCalculationAgent()
        self.target_agent = TargetAchievementAgent()
        self.resource_agent = ResourceOptimizationAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.edl_analyzer = EDLPerformanceAnalyzer(self.metric_agent)
        self.unbilled_analyzer = UnbilledAnalysisAgent(self.metric_agent)
        self.buffer_analyzer = BufferAnalysisAgent(self.metric_agent)

        print("Utilization Analytics Engine Initialized!")

    def _compare_edls(self, edl_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple EDLs and provide comparative analysis"""
        try:
            comparison_data = []

            for edl_analysis in edl_analyses:
                if 'error' not in edl_analysis and 'summary' in edl_analysis:
                    summary = edl_analysis['summary']
                    comparison_data.append({
                        'edl_name': edl_analysis['edl_name'],
                        'overall_utilization': summary.get('overall_utilization', 0),
                        'nbl_utilization': summary.get('nbl_utilization', 0),
                        'total_fte': summary.get('total_fte', 0),
                        'total_projects': summary.get('total_projects', 0),
                        'performance_tier': summary.get('performance_tier', 'UNKNOWN')
                    })

            if comparison_data:
                # Sort by utilization (descending)
                comparison_data.sort(key=lambda x: x['overall_utilization'], reverse=True)

                return {
                    'ranking': comparison_data,
                    'top_performer': comparison_data[0] if comparison_data else None,
                    'needs_attention': [edl for edl in comparison_data if edl['overall_utilization'] < 70],
                    'average_utilization': sum(edl['overall_utilization'] for edl in comparison_data) / len(comparison_data) if comparison_data else 0
                }

            return {}

        except Exception as e:
            print(f"Error comparing EDLs: {e}")
            return {}

    def _compare_edls_detailed(self, edl_names: List[str], prism_data: pd.DataFrame, subcon_data: pd.DataFrame) -> Dict[str, Any]:
        """Detailed comparative analysis between EDLs"""
        try:
            comparison_results = []

            for edl_name in edl_names:
                edl_result = self.edl_analyzer.analyze_edl_performance(edl_name, prism_data, subcon_data)
                comparison_results.append(edl_result)

            # Comparative metrics
            comparison_data = []
            for result in comparison_results:
                if "error" not in result and "summary" in result:
                    summary = result["summary"]
                    comparison_data.append({
                        'edl_name': result.get('edl_name', 'Unknown'),
                        'overall_utilization': summary.get('overall_utilization', 0),
                        'nbl_utilization': summary.get('nbl_utilization', 0),
                        'total_fte': summary.get('total_fte', 0),
                        'total_projects': summary.get('total_projects', 0),
                        'performance_tier': summary.get('performance_tier', 'UNKNOWN')
                    })

            # Generate comparative narrative
            narrative = self._generate_comparative_narrative(comparison_data)

            return {
                'analysis_type': 'edl_comparison',
                'edls_compared': edl_names,
                'comparison_data': comparison_data,
                'narrative_summary': narrative,
                'individual_analyses': comparison_results
            }

        except Exception as e:
            return {"error": f"Comparative analysis error: {str(e)}"}

    def _generate_comparative_narrative(self, comparison_data: List[Dict]) -> str:
        """Generate narrative for EDL comparison"""
        if not comparison_data:
            return "No comparison data available."

        narrative_parts = ["Comparative Analysis:"]

        # Sort by utilization
        sorted_edls = sorted(comparison_data, key=lambda x: x['overall_utilization'], reverse=True)

        narrative_parts.append(f"Ranked by utilization:")
        for i, edl in enumerate(sorted_edls, 1):
            narrative_parts.append(f"{i}. {edl['edl_name']}: {edl['overall_utilization']}% utilization")

        # Performance gaps
        if len(sorted_edls) >= 2:
            gap = sorted_edls[0]['overall_utilization'] - sorted_edls[-1]['overall_utilization']
            narrative_parts.append(f"Performance gap: {gap:.1f}% between top and bottom performer")

        # Size insights
        sizes = [edl['total_fte'] for edl in comparison_data]
        narrative_parts.append(f"Size range: {min(sizes):.1f} to {max(sizes):.1f} FTE")

        return " ".join(narrative_parts)

    def process_query(self, user_query: str) -> Dict[str, Any]:
        print(f"PROCESSING QUERY: '{user_query}'")

        parsed_result = self.llm_parser.parse_query_with_llm(user_query)
        print(f"Parsed intent: {parsed_result['intent']}")
        print(f"Parsed parameters: {parsed_result.get('parameters', {})}")

        prism_data, subcon_data = self.data_agent.get_filtered_data(parsed_result)

        if prism_data.empty:
            return {"error": "No data found for your query scope"}

        # Handle None group_by properly
        raw_group_by = parsed_result.get("group_by")
        if raw_group_by in VALID_GROUPBY_COLUMNS and raw_group_by in prism_data.columns:
            group_by = raw_group_by
            print(f"Grouping by: {group_by}")
        else:
            print(f"Invalid or None group_by: '{raw_group_by}'. Using 'Rev_Mapping'")
            group_by = "Rev_Mapping"

        # Handle different intents
        intent = parsed_result["intent"]
        parameters = parsed_result.get("parameters", {})

        # NEW: Project distribution analysis
        if intent == "project_distribution":
            edl_names = parsed_result.get("entities", {}).get("edl_names", [])
            if edl_names:
                result = self.edl_analyzer.analyze_project_distribution(edl_names[0], prism_data)
            else:
                result = {"error": "No EDL specified for project distribution analysis"}

        # NEW: Location breakdown analysis
        elif intent == "location_breakdown":
            edl_names = parsed_result.get("entities", {}).get("edl_names", [])
            if edl_names:
                result = self.edl_analyzer.analyze_location_breakdown(edl_names[0], prism_data)
            else:
                result = {"error": "No EDL specified for location breakdown analysis"}

        # NEW: TL impact analysis
        elif intent == "tl_impact_analysis":
            scope = parsed_result.get("entities", {}).get("rev_mappings", ["portfolio"])[0]
            result = self.metric_agent.analyze_tl_unbilled_impact(prism_data, scope)

        # NEW: NBL threshold analysis
        elif intent == "nbl_threshold_analysis":
            threshold = parameters.get("nbl_threshold", 25)
            group_by = parsed_result.get("group_by", "Rev_Mapping")
            result = self.diagnostic_agent.analyze_nbl_threshold(prism_data, subcon_data, threshold, group_by)

        # NEW: Comparative analysis (EDL vs EDL)
        elif intent == "comparative_analysis":
            edl_names = parsed_result.get("entities", {}).get("edl_names", [])
            if len(edl_names) >= 2:
                result = self._compare_edls_detailed(edl_names, prism_data, subcon_data)
            else:
                result = {"error": "Need at least 2 EDLs for comparative analysis"}

        # Handle EDL analysis intent - ENHANCED
        elif intent == "edl_analysis":
            edl_names = parsed_result.get("entities", {}).get("edl_names", [])

            # If no specific EDL mentioned, analyze all EDLs
            if not edl_names:
                print("No specific EDL mentioned - analyzing all EDLs")
                edl_names = ["all"]  # Special keyword to trigger all EDLs analysis

            if edl_names[0].lower() == "all":
                # Analyze all EDLs using enhanced method
                result = self.edl_analyzer.analyze_edl_performance("all", prism_data, subcon_data)
            elif len(edl_names) > 1:
                # Multiple specific EDLs
                print(f"Analyzing {len(edl_names)} specific EDLs: {edl_names}")
                all_edl_results = []

                for edl_name in edl_names:
                    print(f"Analyzing EDL: {edl_name}")
                    edl_result = self.edl_analyzer.analyze_edl_performance(edl_name, prism_data, subcon_data)
                    all_edl_results.append(edl_result)

                result = {
                    'query_intent': intent,
                    'analysis_type': 'multi_edl_comparison',
                    'edl_count': len(edl_names),
                    'edl_analyses': all_edl_results,
                    'comparison_metrics': self._compare_edls(all_edl_results),
                    'data_scope': f"Analyzed {len(edl_names)} EDLs",
                    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                # Single EDL analysis using enhanced method
                result = self.edl_analyzer.analyze_edl_performance(edl_names[0], prism_data, subcon_data)

            result["query_intent"] = intent
            result["data_scope"] = f"Analyzed EDL: {edl_names[0] if edl_names and edl_names[0] != 'all' else 'All EDLs'}"
            result["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return result

        # Enhanced unbilled analysis with detailed categorization
        elif intent == "unbilled_analysis":
            scope = parsed_result.get("entities", {}).get("rev_mappings", [])
            scope_name = scope[0] if scope else "portfolio"
            result = self.unbilled_analyzer.analyze_unbilled_resources(prism_data, scope_name)
            # Add detailed category breakdown
            if "error" not in result:
                detailed_breakdown = self.metric_agent.get_detailed_unbilled_analysis(prism_data, scope_name)
                if "error" not in detailed_breakdown and 'category_breakdown' in detailed_breakdown:
                    result["detailed_category_breakdown"] = detailed_breakdown['category_breakdown']
            result["query_intent"] = intent
            result["data_scope"] = f"Analyzed unbilled resources for: {scope_name}"
            result["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return result

        # Enhanced buffer analysis with detailed categorization
        elif intent == "buffer_analysis":
            scope = parsed_result.get("entities", {}).get("rev_mappings", [])
            scope_name = scope[0] if scope else "portfolio"
            result = self.buffer_analyzer.analyze_buffers(prism_data, scope_name)
            # Add detailed category breakdown
            if "error" not in result:
                detailed_breakdown = self.metric_agent.get_detailed_unbilled_analysis(prism_data, scope_name)
                if "error" not in detailed_breakdown and 'category_breakdown' in detailed_breakdown:
                    result["detailed_category_breakdown"] = detailed_breakdown['category_breakdown']
            result["query_intent"] = intent
            result["data_scope"] = f"Analyzed buffers for: {scope_name}"
            result["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return result

        # For other intents, proceed with normal metric calculation
        metrics_df = self.metric_agent.calculate_all_metrics(prism_data, subcon_data, group_by)

        if metrics_df.empty:
            return {"error": "Could not calculate metrics for the filtered data"}

        # Merge forecast targets
        metrics_df = self._merge_forecast_targets(metrics_df, group_by)

        # Handle other intents
        if intent == "target_achievement":
            result = self.target_agent.analyze_achievement_paths(metrics_df, prism_data, self.data_agent.forecast_data, self.data_agent.mapping_data)

        elif intent == "resource_optimization":
            nbl_count = parsed_result.get("parameters", {}).get("nbl_count", 0)
            scope = parsed_result.get("entities", {}).get("rev_mappings", [])
            result = self.resource_agent.optimize_nbl_removal(
                nbl_count, scope, prism_data, subcon_data, self.metric_agent
            )

        elif intent == "diagnostic_analysis":
            analysis_type = parsed_result.get("analysis_type", "unbilled_breakdown")
            result = self.diagnostic_agent.generate_recommendations(metrics_df, prism_data, analysis_type)

        else:
            result = {
                "metrics_summary": self._format_metrics_summary(metrics_df),
                "raw_metrics": metrics_df.to_dict('records')[:10]
            }

        result["query_intent"] = intent
        result["data_scope"] = f"Analyzed {len(metrics_df)} groups"
        result["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        return result

    def _merge_forecast_targets(self, metrics_df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        try:
            print(f"Merging forecast targets for grouping: {group_by}")

            forecast_targets = self.data_agent.get_forecast_targets_by_group(group_by)

            if forecast_targets.empty:
                print("No forecast targets available, using default 70.0%")
                metrics_df['Target_Utilization%'] = 70.0
            else:
                before_merge = len(metrics_df)
                metrics_df = metrics_df.merge(forecast_targets, on=group_by, how='left')
                after_merge = len(metrics_df)

                if 'Target_Utilization%' in metrics_df.columns:
                    nan_count = metrics_df['Target_Utilization%'].isna().sum()
                    if nan_count > 0:
                        overall_avg = forecast_targets['Target_Utilization%'].mean() if not forecast_targets.empty else 70.0
                        print(f"Filling {nan_count} missing targets with overall average: {overall_avg:.1f}%")
                        metrics_df['Target_Utilization%'] = metrics_df['Target_Utilization%'].fillna(overall_avg)
                else:
                    metrics_df['Target_Utilization%'] = 70.0

                print(f"Successfully merged targets for {after_merge} groups (before: {before_merge})")

            metrics_df['Target_Utilization%'] = pd.to_numeric(metrics_df['Target_Utilization%'], errors='coerce').fillna(70.0)
            metrics_df['target_gap_%'] = metrics_df['overall_utilization_%'] - metrics_df['Target_Utilization%']

            target_stats = metrics_df['Target_Utilization%'].describe()
            print(f"Final target utilization stats - Min: {target_stats['min']:.1f}%, Max: {target_stats['max']:.1f}%, Mean: {target_stats['mean']:.1f}%")
            print(f"Target gap stats - Min: {metrics_df['target_gap_%'].min():.1f}%, Max: {metrics_df['target_gap_%'].max():.1f}%")

            return metrics_df

        except Exception as e:
            print(f"Error merging forecast targets: {e}")
            metrics_df['Target_Utilization%'] = 70.0
            metrics_df['target_gap_%'] = metrics_df['overall_utilization_%'] - 70.0
            return metrics_df

    def _format_metrics_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        group_column = next(
            (col for col in VALID_GROUPBY_COLUMNS if col in metrics_df.columns),
            'Group'
        )
        print(f"Using group column: {group_column}")

        return {
            "average_utilization": round(metrics_df['overall_utilization_%'].mean(), 1),
            "average_nbl": round(metrics_df['nbl_utilization_%'].mean(), 1),
            "total_groups": len(metrics_df),
            "top_performers": metrics_df.nlargest(3, 'overall_utilization_%')[[group_column, 'overall_utilization_%']].to_dict('records'),
            "concern_areas": metrics_df.nsmallest(3, 'overall_utilization_%')[[group_column, 'overall_utilization_%']].to_dict('records')
        }
