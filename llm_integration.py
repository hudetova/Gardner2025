"""
Legal Case Analyzer using LLM and Knowledge Graph

This module provides a comprehensive legal analysis system that:
1. Parses legal narratives into chronological events using LLM
2. Determines actor roles for each event based on legal context
3. Evaluates possible legal transitions through a Neo4j knowledge graph
4. Checks requirements recursively using LLM validation
5. Tracks multiple reasoning paths through complex legal scenarios

The analyzer maintains state transitions and provides detailed logging
of the reasoning process for legal analysis and education.
"""

import os
import csv
import pickle
import argparse
from neo4j import GraphDatabase
from collections import Counter
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from typing import Optional, List, Dict, Any, Tuple
from utils import retry_llm_call, TeeOutput


# Loads the environments variables / configurations for the Neo4j and Gemini
load_dotenv(override=True)


class LegalAnalyzerConfig:
    """Configuration constants for the Legal Analyzer"""

    # State and transition constants
    INITIAL_STATE = "NoLegalRelation"
    NO_TRANSITION = "NoTransition"
    FAILED_TRANSITION = "FailedTransition"
    INEFFECTIVE_EVENT_PREFIX = "IneffectiveEvent"

    # LLM Configuration
    MODEL_NAME = "gemini-2.5-flash"
    TEMPERATURE = 0
    MAX_RETRIES = 5
    BASE_DELAY = 1.0
    MAX_DELAY = 60.0

    # Legal roles
    VALID_ROLES = ["Offeror", "Offeree", "Party", "Counterparty", "Third Party"]

    # Display settings
    CONTENT_PREVIEW_LENGTH = 100
    SEPARATOR_LENGTH = 60

    # Logging and output
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_PROGRESS_FILE = "progress.pkl"
    DEFAULT_OUTPUT_FILE = "output.csv"
    DEFAULT_SUMMARY_FILE = "legal_analysis_summary.csv"

    # Database query limits and settings
    INTERIM_STEP_LABEL = "InterimStep"


class LegalEvent(BaseModel):
    actor: str
    action: str
    date: Optional[str] = None
    content: str


class LegalRole(BaseModel):
    legal_role: str
    reasoning: str


class IsRelevantCheck(BaseModel):
    is_relevant: bool
    reasoning: str


class RequirementsCheck(BaseModel):
    requirement_satisfied: bool
    reasoning: str


class LegalAnalysis(BaseModel):
    interpretations: List[str]
    justifications: List[str]


class LegalAnalyzer:
    """
    Legal Case Analyzer using LLM and Knowledge Graph

    This class analyzes legal narratives by:
    1. Parsing narratives into chronological events using LLM
    2. Determining actor roles for each event based on legal context
    3. Evaluating possible legal transitions through a Neo4j knowledge graph
    4. Checking requirements recursively using LLM validation
    5. Tracking multiple reasoning paths through complex legal scenarios

    The analyzer maintains state transitions and provides detailed logging
    of the reasoning process for legal analysis and education.

    Args:
        neo4j_uri (str): URI for Neo4j database connection
        neo4j_user (str): Username for Neo4j authentication
        neo4j_password (str): Password for Neo4j authentication
        gemini_api_key (str): API key for Google Gemini LLM

    Attributes:
        driver: Neo4j database driver instance
        client: Google Gemini API client
        state (str): Current legal state (initialized to INITIAL_STATE)
        model_name (str): Name of the LLM model to use
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, gemini_api_key: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.gemini_api_key = gemini_api_key
        self.state = LegalAnalyzerConfig.INITIAL_STATE
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = LegalAnalyzerConfig.MODEL_NAME

    def analyze(
        self,
        narrative: Optional[str] = None,
        events: Optional[List[LegalEvent]] = None,
        paths: Optional[List[Dict[str, Any]]] = None,
        start_event_num: int = 1,
        auto_save: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[LegalEvent]]:
        """
        Analyze legal narrative or resume from saved progress.

        Args:
            narrative: Legal narrative text to parse (required if events not provided)
            events: Pre-parsed events (for resuming from saved state)
            paths: Current reasoning paths (for resuming from saved state)
            start_event_num: Event number to start/resume from (1-indexed)
            auto_save: Whether to automatically save progress after each event

        Returns:
            Tuple containing:
                - reasoning_paths: List of reasoning paths with their state histories
                - events: List of parsed legal events from the narrative

        Raises:
            ValueError: If neither narrative nor events are provided
        """
        # Handle resuming from saved state vs new analysis
        if events is None:
            if narrative is None:
                raise ValueError("Either narrative or events must be provided")
            # Parse narrative into events (LLM)
            events = self._parse_events(narrative)  # list[LegalEvent]

            print("📋 STARTING NEW ANALYSIS")
            print(f"Found {len(events)} events:")
            for event in events:
                print(f"  • {event.date}: {event.actor} {event.action}")
        else:
            print("📂 RESUMING FROM SAVED STATE")
            print(f"Loaded {len(events)} events, resuming from event {start_event_num}")

        # Initialize paths if not provided
        if paths is None:
            paths = [{"state": LegalAnalyzerConfig.INITIAL_STATE, "history": []}]
            print(f"Starting with clean initial state: {LegalAnalyzerConfig.INITIAL_STATE}")
        else:
            print(f"Continuing with {len(paths)} existing reasoning paths")

        # Process each event starting from start_event_num
        for event_num, event in enumerate(events, 1):
            # Skip events before start_event_num (for resuming)
            if event_num < start_event_num:
                continue
            print(f"\n{'='*60}")
            print(f"📅 PROCESSING EVENT {event_num} of {len(events)}")
            print(f"{'='*60}")
            print(f"Date: {event.date}")
            print(f"Actor: {event.actor}")
            print(f"Action: {event.action}")
            print(f"Content: {event.content}")
            print(f"{'='*60}")

            new_paths = []

            for path_num, path in enumerate(paths, 1):
                # Check if path was terminated by failed transition
                if path.get("terminated", False):
                    print(
                        f"\n⚠️ Path {path_num} of {len(paths)}: "
                        f"This path terminated at Event {len(path['history'])}."
                    )
                    print("   " + "-" * 60)

                    # Add to new_paths unchanged to preserve for final analysis
                    new_paths.append(path)
                    continue  # Skip to next path

                ############## START LOGGING ##############  # noqa
                print(f"\n👣 Path {path_num} of {len(paths)}: {LegalAnalyzerConfig.INITIAL_STATE}")

                # Track the current state as we iterate through history
                current_state = LegalAnalyzerConfig.INITIAL_STATE

                for event_record in path["history"]:
                    transition = event_record["transition"]

                    # Determine next state after this transition
                    if transition in [
                        LegalAnalyzerConfig.NO_TRANSITION,
                        LegalAnalyzerConfig.FAILED_TRANSITION,
                    ]:
                        # FailedTransition doesn't change the state
                        next_state = current_state
                    else:
                        # For successful transitions, check if this is the last event
                        event_idx = path["history"].index(event_record)
                        if event_idx < len(path["history"]) - 1:
                            # Not the last event - peek at next event's starting state
                            next_state = path["history"][event_idx + 1]["state"]
                        else:
                            # This is the last event - use path's current state
                            next_state = path["state"]

                    print(f"   → {transition} == {next_state}")
                    current_state = next_state
                ############## END LOGGING ##############

                # Determine current role of the actor
                actor_role, actor_role_reasoning = self._get_actor_role(
                    event, path["state"], path["history"]
                )

                ############## START LOGGING ##############
                # Log actor role assignment
                print("\n🎭 ACTOR ROLE ASSIGNMENT:")
                print(f"   Actor: {event.actor}")
                print(f"   Action: {event.action}")
                print(f"   Current State: {path['state']}")
                print(f"   Assigned Role: {actor_role}")
                print(f"   Explanation: {actor_role_reasoning}")
                ############## END LOGGING ##############

                # Get possible transitions from the current state, for the given actor role
                results = self._get_possible_transitions(path["state"], actor_role)

                # If no possible transitions found for this state and role.
                if len(results) == 0:
                    print(f"   ⚫ NO VALID TRANSITIONS FOUND for {event.actor}'s {event.action}")
                    print(f"   🔄 State remains: {path['state']}")

                    # Add to the reasoning path a NoTransition
                    event_record = {
                        "num": event_num,
                        "transition": LegalAnalyzerConfig.NO_TRANSITION,
                        "state": path["state"],
                        "actor": event.actor,
                        "action": event.action,
                        "evidence": event.content,
                        "checks": [],
                        "reason": "NoTransition - No valid transitions available for this state and role",
                    }
                    new_path = {
                        "state": path["state"],
                        "history": path["history"] + [event_record],
                    }
                    new_paths.append(new_path)
                    continue

                ############## START LOGGING ##############
                print("\n🔍 POSSIBLE TRANSITIONS FROM KNOWLEDGE GRAPH:")
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. {result['name']} → {result['next_state']}")
                        print(f"      Role requirement: {result['by']}")
                        if result["details"]:
                            print(f"      Details: {result['details']}")
                else:
                    print("   No possible transitions found for this state and role.")
                ############## END LOGGING ##############

                ############## START LOGGING ##############
                print("\n⏳  EVALUATING TRANSITIONS:")
                ############## END LOGGING ##############

                successful_transitions = []  # Track successful transitions that pass all checks

                # From the current state, try each possible transition, argue for it
                for transition_num, result in enumerate(results, 1):
                    print(f"\n   --- TRANSITION {transition_num}: {result['name']} ---")

                    # Quick semantic filter to skip irrelevant arcs (LLM)
                    is_relevant, reason = self._is_relevant(
                        result["name"], event, output_schema=IsRelevantCheck
                    )

                    ############## START LOGGING ##############
                    if is_relevant:
                        print(f"1️⃣   RELEVANCE CHECK: ✅ RELEVANT. {reason}")
                    else:
                        print(
                            f"1️⃣   RELEVANCE CHECK: ❌ NOT RELEVANT. {reason}. "
                            f"Skipping transition '{result['name']}'"
                        )
                    ############## END LOGGING ##############

                    # If the transition is not relevant, skip it
                    if not is_relevant:
                        continue

                    # For each relevant arc, check requirements RECURSIVELY (LLM at leaves)
                    all_met, recursive_checks = self._check_requirement_recursively(
                        result["name"],
                        result.get("details"),
                        event,
                        path["history"],
                        path["state"],
                        actor_role,
                    )

                    ############## START LOGGING ##############
                    # Print hierarchical requirements immediately after checking this transition
                    print(f"2️⃣   REQUIREMENT CHECKS:")
                    if recursive_checks:
                        print(f"      Requirements for '{result['name']}':")
                        self._print_checks_hierarchically(recursive_checks, indent=3)
                    else:
                        print(f"      No requirements to check for '{result['name']}'")
                    ############## END LOGGING ##############

                    # Only create event records and paths for SUCCESSFUL transitions
                    if all_met:
                        event_record = {
                            "num": event_num,
                            "transition": result["name"],
                            "state": path["state"],  # kept for the summary table
                            "actor": event.actor,
                            "action": event.action,
                            "evidence": event.content,
                            "checks": recursive_checks,
                            "reason": reason,
                        }

                        new_path = {
                            "state": result["next_state"],
                            "history": path["history"] + [event_record],
                        }
                        successful_transitions.append(
                            (transition_num, result, new_path, event_record)
                        )

                    ############## START LOGGING ##############
                    if all_met:
                        print(
                            f"      → RESULT: ✅ ARGUMENT PASSED. ➡️  Transition to: {result['next_state']}"
                        )
                    else:
                        print(f"      → RESULT: ❌ ARGUMENT FAILED.")
                    ############## END LOGGING ##############

                ############## START COUNTEROFFER-REJECTION PRUNING ##############
                # Filter out redundant rejection paths when counteroffer exists
                filtered_transitions = []
                has_counteroffer = False
                has_rejection = False

                # Check what types of transitions we have
                for transition_num, result, new_path, event_record in successful_transitions:
                    if "counteroffer" in result["name"].lower():
                        has_counteroffer = True
                    if "rejection" in result["name"].lower():
                        has_rejection = True

                # If we have both counteroffer and rejection, drop the rejection paths
                if has_counteroffer and has_rejection:
                    print("\n⚠️ COUNTEROFFER-REJECTION PRUNING:")
                    print("   Found both counteroffer and rejection transitions")
                    print("   Counteroffer inherently includes rejection. Rejection is redundant.")

                    for transition_num, result, new_path, event_record in successful_transitions:
                        if "rejection" not in result["name"].lower():
                            filtered_transitions.append(
                                (transition_num, result, new_path, event_record)
                            )
                            new_paths.append(new_path)
                        else:
                            print(f"   ❌ Pruned redundant path: {result['name']}")
                else:
                    # No pruning needed - keep all successful transitions
                    for transition_num, result, new_path, event_record in successful_transitions:
                        filtered_transitions.append(
                            (transition_num, result, new_path, event_record)
                        )
                        new_paths.append(new_path)

                # Update successful_transitions to only include non-pruned transitions
                successful_transitions = [(t[0], t[1]) for t in filtered_transitions]
                ############## END COUNTEROFFER-REJECTION PRUNING ##############

                # If no transitions succeeded at all (failed requirements), path continues unchanged
                if len(successful_transitions) == 0 and len(results) > 0:
                    print(
                        f"   NO TRANSITIONS PASSED REQUIREMENTS for {event.actor}'s {event.action}"
                    )
                    print(f"   🔄 State remains: {path['state']}")

                    # Add to the reasoning path a NoTransition
                    event_record = {
                        "num": event_num,
                        "transition": LegalAnalyzerConfig.NO_TRANSITION,
                        "state": path["state"],
                        "actor": event.actor,
                        "action": event.action,
                        "evidence": event.content,
                        "checks": [],
                        "reason": "No transitions passed requirements",
                    }
                    new_path = {
                        "state": path["state"],
                        "history": path["history"] + [event_record],
                    }
                    new_paths.append(new_path)

                ############## START LOGGING ##############
                print("\n⏳  EVALUATING ARGUMENTS AGAINST TRANSITIONS:")
                print(f"   Number of successful transitions: {len(successful_transitions)}")
                ############## END LOGGING ##############

                successfully_argued_against_transitions = []
                recursive_negation_checks = []
                for transition_num, result in enumerate(results, 1):

                    # Only proceeed if it is one of the successful transitions
                    if transition_num not in [t[0] for t in successful_transitions]:
                        continue

                    ############## START LOGGING ##############
                    print(
                        f"\n   --- ARGUMENTING AGAINST TRANSITION {transition_num}: {result['name']} ---"
                    )
                    ############## END LOGGING ##############

                    # Check all the requirements of this transition and argue against them
                    # then see if the negation of the logical composition of the requirements is
                    # satisfied
                    negation_met, recursive_negation_checks = (
                        self._check_counter_arguments_recursively(
                            result["name"],
                            result.get("details"),
                            event,
                            path["history"],
                            actor_role,
                        )
                    )

                    ############## START LOGGING ##############
                    # Print hierarchical counter-argument checks immediately after checking this transition
                    print(f"3️⃣   COUNTER-ARGUMENT CHECKS:")
                    if recursive_negation_checks:
                        print(f"      Counter-arguments for '{result['name']}':")
                        self._print_checks_hierarchically(
                            recursive_negation_checks, indent=3, arguing_against=True
                        )
                    else:
                        print(f"      No counter-arguments to check for '{result['name']}'")
                    ############## END LOGGING ##############

                    if negation_met:
                        successfully_argued_against_transitions.append((transition_num, result))

                    ############## START LOGGING ##############
                    if negation_met:
                        print("      → RESULT: COUNTER-ARGUMENT PASSED. ❌ ARGUMENT DEFEATED.")
                    else:
                        print("      → RESULT: COUNTER-ARGUMENT FAILED. ✅ ARGUMENT STANDS.")
                    ############## END LOGGING ##############

                # If we could argue against all the successful transitions, add a FailedTransition to the reasoning path
                if len(successful_transitions) > 0 and len(
                    successfully_argued_against_transitions
                ) == len(successful_transitions):
                    # Check if this is an offer from NoLegalRelation state
                    is_initial_offer = path["state"] == LegalAnalyzerConfig.INITIAL_STATE and any(
                        "offer" in t[1]["name"].lower()
                        for t in successfully_argued_against_transitions
                    )

                    event_record = {
                        "num": event_num,
                        "transition": LegalAnalyzerConfig.FAILED_TRANSITION,
                        "state": path["state"],
                        "actor": event.actor,
                        "action": event.action,
                        "evidence": event.content,
                        "checks": recursive_negation_checks,
                        "reason": "FailedTransition - All successful transitions were defeated by counter-arguments",
                    }

                    new_path = {
                        "state": path["state"],  # State remains unchanged
                        "history": path["history"] + [event_record],
                    }

                    # Only terminate if NOT an initial offer
                    if not is_initial_offer:
                        print("\n⚠️  COUNTER-ARGUMENT ANALYSIS SHOWS FAILED TRANSITION RISK:")
                        print(
                            f"   Counter-argument can defeat all successful transitions ({len(successful_transitions)})."
                        )
                        print(
                            f"   In this case: Event '{event.actor} {event.action}' has no legal effect. Transition fails. 🔄 State remains: {path['state']}."
                        )
                        print("   " + "-" * 80)
                        print(
                            "   Failed transitions are not analysed further. Branching path terminates here."
                        )
                        print("   " + "-" * 80)

                        new_path["terminated"] = True
                        new_path["termination_reason"] = "All legal arguments defeated"

                    new_paths.append(new_path)

            paths = new_paths

            # Summarize the analysis after current event
            print(
                f"\n>>> EVENT {event_num} OF {len(events)} COMPLETED: {event.actor} {event.action}"
            )
            print(f"Event content: {event.content}")

            active_paths = [p for p in paths if not p.get("terminated", False)]
            terminated_paths = [p for p in paths if p.get("terminated", False)]

            if terminated_paths:
                print(
                    f"▶️  ACTIVE PATHS: {len(active_paths)} | ⚠️  TERMINATED PATHS: {len(terminated_paths)}"
                )
            else:
                print(f"Resulting paths: {len(paths)}")

            for i, path in enumerate(paths):
                if path.get("terminated", False):
                    print(
                        f"⚠️  [Terminated] Path {i+1}: Counter-argument can defeat all successful transitions."
                    )
                else:
                    print(f"👣 Path {i+1}: {LegalAnalyzerConfig.INITIAL_STATE}")

                    # Track the current state as we iterate through history
                    current_state = LegalAnalyzerConfig.INITIAL_STATE

                    for event_record in path["history"]:
                        transition = event_record["transition"]

                        # Determine next state after this transition
                        if transition in [
                            LegalAnalyzerConfig.NO_TRANSITION,
                            LegalAnalyzerConfig.FAILED_TRANSITION,
                        ]:
                            # Failed transitions don't change the state
                            next_state = current_state
                        else:
                            # For successful transitions, check if this is the last event
                            event_idx = path["history"].index(event_record)
                            if event_idx < len(path["history"]) - 1:
                                # Not the last event - peek at next event's starting state
                                next_state = path["history"][event_idx + 1]["state"]
                            else:
                                # This is the last event - use path's current state
                                next_state = path["state"]

                        print(f"   → {transition} == {next_state}")
                        current_state = next_state

            print(f">>> RECORDING EVENT {event_num} OF {len(events)}")

            # Auto-save progress after each event
            try:
                save_info = self.save_progress(events, paths, event_num)
                print(f"💾 Event {event_num} auto-saved: {save_info['pickle_file']}")
            except Exception as e:
                print(f"⚠️  Auto-save failed: {e}")

            # input("Press Enter to continue to next event...")

        # Print final results with statistics on the possible states distribution after each event
        self.generate_summary_table(events, paths)
        return paths, events

    def generate_summary_table(
        self,
        events: List[LegalEvent],
        reasoning_paths: List[Dict[str, Any]],
        output_path: str = LegalAnalyzerConfig.DEFAULT_OUTPUT_FILE,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate a comprehensive summary table of all reasoning paths and state distributions.

        Args:
            events: List of events returned by analyze()
            reasoning_paths: List of reasoning paths returned by analyze()
            output_path: Path where to save the CSV file

        Returns:
            Dictionary mapping event numbers to state distribution statistics
        """
        if not reasoning_paths:
            print("No reasoning paths to analyze.")
            return {}

        # Prepare CSV data
        csv_data = []
        state_distributions = {}

        # Create header row
        header = ["Event_Num", "Actor", "Action", "Evidence"]
        for i in range(len(reasoning_paths)):
            header.append(f"Path_{i+1}")
        csv_data.append(header)

        # Process each event
        for event_num in range(len(events)):
            event = events[event_num]
            row = [
                event_num + 1,
                event.actor,
                event.action,
                event.content,
            ]

            # Add data for each reasoning path
            states_after_event = []
            for path in reasoning_paths:
                event_state_in_path = path["history"][event_num]["state"]
                states_after_event.append(event_state_in_path)
            row.extend(states_after_event)
            csv_data.append(row)

            # Calculate state distribution after this event
            state_counts = Counter(states_after_event)
            state_distributions[event_num] = {
                "total_paths": len(states_after_event),
                "distribution": state_counts,
            }

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        # Print summary
        print(f"\n📊 REASONING PATHS SUMMARY")
        print(f"{'='*60}")
        print(f"Total reasoning paths analyzed: {len(reasoning_paths)}")
        print(f"Total events processed: {len(events)}")
        print(f"Results saved to: {output_path}")

        print("\n🎯 STATE DISTRIBUTIONS AFTER EACH EVENT")
        print(f"{'='*60}")
        for event_num, stats in state_distributions.items():
            print(f"\nAfter Event {event_num + 1}:")
            print("  State distribution:")
            for state, counts in stats["distribution"].items():
                print(f"    • {state}: {counts} paths")

        return state_distributions

    def save_progress(
        self,
        events: List[LegalEvent],
        paths: List[Dict[str, Any]],
        event_num: int,
        save_dir: str = LegalAnalyzerConfig.DEFAULT_LOG_DIR,
    ) -> Dict[str, Any]:
        """
        Save the current progress of the reasoning analysis to files.

        Args:
            events: List of events being processed
            paths: Current reasoning paths
            event_num: Current event number being processed (1-indexed)
            save_dir: Directory to save the progress files

        Returns:
            Dictionary containing information about saved files including paths and metadata
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save comprehensive state as pickle
        state_data = {
            "events": events,
            "paths": paths,
            "event_num": event_num,
            "total_events": len(events),
            "total_paths": len(paths),
        }
        event_progress_file = f"{LegalAnalyzerConfig.DEFAULT_PROGRESS_FILE}_{event_num}.pkl"
        pickle_file = os.path.join(save_dir, event_progress_file)
        with open(pickle_file, "wb") as f:
            pickle.dump(state_data, f)

        return {
            "pickle_file": pickle_file,
            "event_num": event_num,
            "total_events": len(events),
            "total_paths": len(paths),
        }

    @staticmethod
    def load_progress(
        pickle_file: str = f"{LegalAnalyzerConfig.DEFAULT_LOG_DIR}/{LegalAnalyzerConfig.DEFAULT_PROGRESS_FILE}",
        event_num: Optional[int] = None,
    ) -> Tuple[List[LegalEvent], List[Dict[str, Any]], int]:
        """
        Load previously saved reasoning progress from a pickle file.

        Args:
            pickle_file: Path to the pickle file containing saved progress
            event_num: Event number to load progress for
        Returns:
            Tuple containing:
                - events: List of parsed legal events
                - paths: List of reasoning paths with state histories
                - event_num: Last processed event number (1-indexed)

        Raises:
            FileNotFoundError: If the progress file does not exist
        """

        if event_num is not None:
            pickle_file = f"{LegalAnalyzerConfig.DEFAULT_LOG_DIR}/{LegalAnalyzerConfig.DEFAULT_PROGRESS_FILE}_{event_num}.pkl"

        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Progress file not found: {pickle_file}")

        with open(pickle_file, "rb") as f:
            state_data = pickle.load(f)

        events = state_data["events"]
        paths = state_data["paths"]
        event_num = state_data["event_num"]

        print("\n📂 PROGRESS LOADED:")
        print(f"   File: {pickle_file}")
        print(f"   Event: {event_num}/{len(events)}")
        print(f"   Paths: {len(paths)}")

        return events, paths, event_num

    def _print_checks_hierarchically(
        self, checks: Optional[List[Dict[str, Any]]], indent: int = 0, arguing_against: bool = False
    ) -> None:
        """
        Print requirement checks in a hierarchical tree structure.

        Args:
            checks: List of check dictionaries containing term, met status, logic, and explanation
            indent: Current indentation level for nested display
            arguing_against: Whether the checks are for counter-arguments
        """
        for check in checks or []:
            prefix = "  " * indent

            if arguing_against:
                status = "🔴" if check.get("met", False) else "🟢"
            else:
                status = "🟢" if check.get("met", False) else "🔴"

            term = check.get("term", "")
            logic = check.get("logic", "")
            explanation = check.get("explanation", "")

            print(f"{prefix}{status} {term} ({logic}) {explanation}")

            # Recursively print children with increased indentation
            if check.get("children"):
                self._print_checks_hierarchically(check["children"], indent + 1, arguing_against)

    def _check_requirement_recursively(
        self,
        term: str,
        details: Optional[str],
        event: LegalEvent,
        history: Optional[List[Dict[str, Any]]] = None,
        state: Optional[str] = None,
        actor_role: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Recursively checks requirements from the knowledge graph.

        This method follows REQUIRES edges using AND/OR logic and uses LLM
        evaluation only at terminal nodes to determine if requirements are satisfied.

        Args:
            term: The requirement term to check
            details: Additional details about the requirement
            event: Current legal event being evaluated
            history: List of past events for context in terminal requirement evaluation
            state: Current state of the reasoning path
            actor_role: The role assigned to the actor for this specific path

        Returns:
            Tuple containing:
                - bool: Whether the requirement is satisfied
                - List[Dict]: Hierarchical structure of requirement checks
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {term: $term})
                OPTIONAL MATCH (n)-[r:REQUIRES]->(req)
                RETURN n.sense as current_sense, 
                    req.term as term, 
                    r.logic as logic, 
                    req.sense as details
                """,
                term=term,
            ).data()

        if not result or all(r["term"] is None for r in result):  # Terminal node
            current_sense = result[0]["current_sense"] if result else details
            met, explanation = self._evaluate_terminal_requirement(
                term, current_sense, event, history, state, actor_role
            )
            return met, [{"term": term, "met": met, "logic": "LEAF", "explanation": explanation}]

        # Filter out null requirements and get current sense
        current_sense = result[0]["current_sense"] if result else details
        requirements = [r for r in result if r["term"] is not None]

        and_reqs = [r for r in requirements if r["logic"] == "AND"]
        or_reqs = [r for r in requirements if r["logic"] == "OR"]

        child_checks = []

        all_and_met = True
        for req in and_reqs:
            sub_met, sub_checks = self._check_requirement_recursively(
                req["term"], req["details"], event, history, state, actor_role
            )
            child_checks.extend(sub_checks)
            if not sub_met:
                all_and_met = False

        any_or_met = not or_reqs
        if or_reqs:
            or_met_found = False
            for req in or_reqs:
                sub_met, sub_checks = self._check_requirement_recursively(
                    req["term"], req["details"], event, history, state, actor_role
                )
                child_checks.extend(sub_checks)
                if sub_met:
                    or_met_found = True
            if or_met_found:
                any_or_met = True

        overall_met = all_and_met and any_or_met
        top_level_check = {
            "term": term,
            "met": overall_met,
            "logic": (
                "AND" if and_reqs and not or_reqs else "OR" if or_reqs and not and_reqs else "MIXED"
            ),
            "children": child_checks,
        }
        return overall_met, [top_level_check]

    def _check_counter_arguments_recursively(
        self,
        term: str,
        details: Optional[str],
        event: LegalEvent,
        history: Optional[List[Dict[str, Any]]] = None,
        state: Optional[str] = None,
        actor_role: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Recursively checks counter-arguments from the knowledge graph.

        This method follows REQUIRES edges using inverted AND/OR logic for negation
        and uses LLM only at terminal nodes to argue against requirement satisfaction.

        Negation logic:
        - NOT(A AND B) = (NOT A) OR (NOT B) - at least one negation must succeed
        - NOT(A OR B) = (NOT A) AND (NOT B) - all negations must succeed

        Args:
            term: The requirement term to check counter-arguments for
            details: Additional details about the requirement
            event: Current legal event being evaluated
            history: List of past events for context in terminal requirement evaluation
            state: Current state of the reasoning path
            actor_role: The role assigned to the actor for this specific path
        Returns:
            Tuple containing:
                - bool: Whether counter-arguments successfully negate the requirement
                - List[Dict]: Hierarchical structure of counter-argument checks
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {term: $term})
                OPTIONAL MATCH (n)-[r:REQUIRES]->(req)
                RETURN n.sense as current_sense, 
                    req.term as term, 
                    r.logic as logic, 
                    req.sense as details
                """,
                term=term,
            ).data()

        if not result or all(r["term"] is None for r in result):  # Terminal node
            current_sense = result[0]["current_sense"] if result else details
            not_met, explanation = self._evaluate_terminal_requirement_counter(
                term, current_sense, event, history, state, actor_role
            )
            return not_met, [
                {"term": term, "met": not_met, "logic": "LEAF", "explanation": explanation}
            ]

        # Filter out null requirements and get current sense
        current_sense = result[0]["current_sense"] if result else details
        requirements = [r for r in result if r["term"] is not None]

        and_reqs = [r for r in requirements if r["logic"] == "AND"]
        or_reqs = [r for r in requirements if r["logic"] == "OR"]

        child_checks = []

        # For counter-arguments (negation):
        # NOT(A AND B) = (NOT A) OR (NOT B) - at least one negation must succeed
        # NOT(A OR B) = (NOT A) AND (NOT B) - all negations must succeed

        # Handle AND requirements (negation: OR of negations - at least one must fail)
        any_and_negated = False
        for req in and_reqs:
            sub_not_met, sub_checks = self._check_counter_arguments_recursively(
                req["term"], req["details"], event, history, state, actor_role
            )
            child_checks.extend(sub_checks)
            if sub_not_met:  # If we can argue against at least one AND requirement
                any_and_negated = True

        # Handle OR requirements (negation: AND of negations - all must fail)
        all_or_negated = True  # If no OR requirements, this condition is satisfied (vacuous truth)
        if or_reqs:
            for req in or_reqs:
                sub_not_met, sub_checks = self._check_counter_arguments_recursively(
                    req["term"], req["details"], event, history, state, actor_role
                )
                child_checks.extend(sub_checks)
                if not sub_not_met:  # If we cannot argue against this OR requirement
                    all_or_negated = False

        # Overall negation succeeds if:
        # - We can negate at least one AND requirement (if any exist), AND
        # - We can negate all OR requirements (if any exist)
        overall_negated = (not and_reqs or any_and_negated) and all_or_negated

        top_level_check = {
            "term": term,
            "met": overall_negated,
            "logic": (
                "NOT_AND"
                if and_reqs and not or_reqs
                else "NOT_OR" if or_reqs and not and_reqs else "NOT_MIXED"
            ),
            "children": child_checks,
        }
        return overall_negated, [top_level_check]

    def _evaluate_terminal_requirement(
        self,
        requirement: str,
        details: Optional[str],
        event: LegalEvent,
        history: Optional[List[Dict[str, Any]]] = None,
        state: Optional[str] = None,
        actor_role: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Use LLM to check if a terminal requirement is satisfied by the current event.

        Args:
            requirement: The legal requirement term to evaluate
            details: Additional context or explanation about the requirement
            event: Current legal event being evaluated
            history: List of past events for context in requirement evaluation
            state: Current state of the reasoning path
            actor_role: The role assigned to the actor for this event in this path
        Returns:
            Tuple containing:
                - bool: Whether the requirement is satisfied
                - str: Reasoning explanation from the LLM
        """

        # Format history for context if available
        history_context = ""
        if history and len(history) > 0:
            history_context = "\n\nPrevious Events and Legal Outcomes:\n"
            for i, past_event in enumerate(history, 1):
                transition = past_event.get("transition", "No transition")
                actor = past_event.get("actor", "Unknown")
                action = past_event.get("action", "performed action")
                evidence = past_event.get("evidence", "No details")

                history_context += f"{i}. {actor} {action}: {evidence}\n"
                if not LegalAnalyzer.is_failed_transition(transition):
                    history_context += f"   → Legal outcome: {transition} was established\n"
                else:
                    history_context += f"   → Legal outcome: No legal effect\n"
                history_context += "\n"

            history_context += f"CURRENT LEGAL STATE: {state}\n"
            history_context += "Note: The current legal state reflects what legal elements have been established through previous events. "
            history_context += "When evaluating requirements, consider both the current event AND the legal context that led to this state.\n\n"

        # Define a unified set of guidelines for both argument and counter-argument analysis.
        # This ensures consistency in reasoning, with the only variable being the advocate's goal.
        unified_guidelines = """
            ## Guidelines for Your Analysis
            1.  **Focus on Argumentation:** Look for legal, factual, or procedural reasons for your task.
            2.  **Ground Arguments in Facts:** Base your reasoning strictly on the details provided in the 'Current Event' and the 'Legal Rule'. 
            3.  **Distinguish Active vs. Passive Actions:** The nature of the 'Event Action' is critical. An actor performing a PASSIVE action (e.g., "received a telegram") cannot satisfy a rule that requires an ACTIVE deed (e.g., making an offer, accepting). If the action itself is passive, the content of a act/communication was done by someone else.
                - Passive actions (e.g., "received", "has seen", "was notified") describe things that happen TO a party, not actions BY a party.
                - Active events (e.g., "sent", "decided", "responded") describe actions that a party deliberately performs.
                - IMPORTANT: Sending content can mean either initiating an action or responding to a previous action.
                - IMPORTANT: Receiving content does NOT mean performing the action described in that content.
                - Example: "Buyer received Seller's acceptance" means Seller accepted (active), Buyer was informed (passive). The Buyer did NOT accept anything.
            4.  **Prioritize Direct Interpretation:** Prefer the most direct interpretation of an event. Avoid complex, secondary inferences if a more direct interpretation is available.
            5.  **Acknowledge Ambiguity:** If the facts are ambiguous, explain the ambiguity but conclude based on whether a plausible argument can be constructed from those facts that aligns with your goal (either satisfying or not satisfying the rule).                
            6. Keep your reasoning short, concise, straightforward and to the point, while containing all the information needed to make a decision.

        """

        requirements_check_prompt = f"""
            You are an expert legal analyst. Your task is to determine if a plausible affirmative argument can be constructed that the 'Current Event' satisfies the 'Legal Rule'.

            You must adopt the mindset of a lawyer building a case. Your goal is to find a credible argument, even if it's not guaranteed to win in court. You are not a neutral academic or judge; you are an advocate for the position that the rule is satisfied.

            ## Context
            - Previous Events & Outcomes: {history_context}
            - Current Legal State: {state}
            - Current Actor's Role for this Path: {actor_role}
            - Current Event to Evaluate: {event.content}
            - Event Action: {event.action}

            ## Your Task
            - Legal Rule: {requirement} ({details})
            {unified_guidelines}
            ## Output Requirements
            - Output in JSON format.
            - Provide a 'reasoning' field explaining how a plausible argument could be constructed.
            - Provide a 'requirement_satisfied' field (true/false).
            """

        response = self._llm(requirements_check_prompt, output_schema=RequirementsCheck)
        return response.requirement_satisfied, response.reasoning

    def _evaluate_terminal_requirement_counter(
        self,
        requirement: str,
        details: Optional[str],
        event: LegalEvent,
        history: Optional[List[Dict[str, Any]]] = None,
        state: Optional[str] = None,
        actor_role: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Use LLM to check if we can argue that a terminal requirement is NOT satisfied.

        Args:
            requirement: The legal requirement term to evaluate counter-arguments for
            details: Additional context or explanation about the requirement
            event: Current legal event being evaluated
            history: List of past events for context in requirement evaluation
            state: Current state of the reasoning path
            actor_role: The role assigned to the actor for this event in this path
        Returns:
            Tuple containing:
                - bool: Whether we can successfully argue the requirement is not satisfied
                - str: Counter-argument reasoning explanation from the LLM
        """

        # Format history for context if available (same as original)
        history_context = ""
        if history and len(history) > 0:
            history_context = "\n\nPrevious Events and Legal Outcomes:\n"
            for i, past_event in enumerate(history, 1):
                transition = past_event.get("transition", "No transition")
                actor = past_event.get("actor", "Unknown")
                action = past_event.get("action", "performed action")
                evidence = past_event.get("evidence", "No details")

                history_context += f"{i}. {actor} {action}: {evidence}\n"
                if not LegalAnalyzer.is_failed_transition(transition):
                    history_context += f"   → Legal outcome: {transition} was established\n"
                else:
                    history_context += f"   → Legal outcome: No legal effect\n"
                history_context += "\n"

            history_context += f"CURRENT LEGAL STATE: {state}\n"
            history_context += "Note: The current legal state reflects what legal elements have been established through previous events. "
            history_context += "When evaluating requirements, consider both the current event AND the legal context that led to this state.\n\n"

        # Define a unified set of guidelines for both argument and counter-argument analysis.
        # This ensures consistency in reasoning, with the only variable being the advocate's goal.
        unified_guidelines = """
            ## Guidelines for Your Analysis
            1.  **Ground Arguments in Facts:** Base your reasoning strictly on the details provided in the 'Current Event' and the 'Legal Rule'.            
            3.  **Prioritize Direct Interpretation:** Prefer the most direct interpretation of an event. Avoid complex, secondary inferences if a more direct interpretation is available.
            2.  **Distinguish Active vs. Passive Actions:** The nature of the 'Event Action' is critical. An actor performing a PASSIVE action (e.g., "received a telegram") cannot satisfy a rule that requires an ACTIVE deed (e.g., making an offer, accepting). The content of a communication is irrelevant if the action itself is passive.
            4.  **Acknowledge Ambiguity:** If the facts are ambiguous, explain the ambiguity but conclude based on whether a plausible argument can be constructed from those facts that aligns with your goal (either satisfying or not satisfying the rule).
        """

        counter_requirements_check_prompt = f"""
            You are an expert legal analyst. Your task is to determine if a plausible counter-argument can be constructed that the 'Current Event' does NOT satisfy the 'Legal Rule'.

            You must adopt the mindset of an opposing counsel looking for flaws in an argument. Your goal is to find any credible reason—legal, factual, or procedural—why the rule might not be met. You are not a neutral academic or judge; you are an advocate for the position that the rule is NOT satisfied.

            ## Context
            - Previous Events & Outcomes: {history_context}
            - Current Legal State: {state}
            - Current Actor's Role for this Path: {actor_role}
            - Legal Rule Being Challenged: {requirement} ({details})
            - Current Event to Evaluate: {event.content}
            - Event Action: {event.action}

            ## Your Task
            - Analyze the event and determine if a plausible counter-argument exists.
            {unified_guidelines}
            ## Output Requirements
            - Output in JSON format.
            - Provide a 'reasoning' field explaining how a plausible counter-argument could be constructed.
            - Provide a 'requirement_satisfied' field (true if a counter-argument was found, false otherwise).
            """

        response = self._llm(counter_requirements_check_prompt, output_schema=RequirementsCheck)
        return response.requirement_satisfied, response.reasoning

    def _parse_events(self, narrative: str) -> List[LegalEvent]:
        """
        Parse a legal narrative into chronological events using LLM.

        Args:
            narrative: Legal narrative text to parse

        Returns:
            List of parsed LegalEvent objects in chronological order
        """
        extract_events_prompt = f"""
            Extract ALL legal events from this narrative in chronological order.
            
            For each event, identify:
            - ACTOR: Who performed the action
            - ACTION: What they did
            - DATE: When they did it (or "unknown" if not specified)
            - ROLE: Assign the most specific legal role that applies to the actor:
                -- Use specific roles when clear (offeror, offeree, trustee, etc.)
                -- Fall back to universal roles (party, counterparty, third party, etc.) when specific role is unclear
                -- For the first actor in a sequence, typically use "party"
            - CONTENT: The complete text/description of what was communicated or done
            
            IMPORTANT: Your only task is to faithfully extract events from the narrative. Therefore, your extraction MUST be strictly grounded in the words of the narrative. Do NOT infer legal meaning. All legal analysis will be performed by a different part of the system.

            IMPORTANT: Return events in the actual chronological order they occurred.

            NARRATIVE: {narrative}
        """
        events = self._llm(extract_events_prompt, output_schema=list[LegalEvent])
        for event in events:
            if not hasattr(event, "date") or event.date is None:
                event.date = "unknown"

        return events

    def _is_relevant(
        self, arc_name: str, event: LegalEvent, output_schema: type = IsRelevantCheck
    ) -> Tuple[bool, str]:
        """
        Quick LLM check to determine if a transition arc applies to the current event.

        Args:
            arc_name: Name of the legal transition arc to check
            event: Current legal event being evaluated
            output_schema: Pydantic model for structured LLM output

        Returns:
            Tuple containing:
                - bool: Whether the arc is relevant to the event
                - str: Reasoning explanation from the LLM
        """

        is_relevant_prompt = f"""
        Could the action "{event.action}" possibly be related to "{arc_name}"?
        Answer YES if there's any possible connection.
        Answer NO only if clearly unrelated (like Death when someone sends a telegram).

        Output in JSON format.
        """
        response = self._llm(is_relevant_prompt, output_schema=IsRelevantCheck)
        is_relevant = response.is_relevant
        reasoning = response.reasoning
        return is_relevant, reasoning

    def _get_possible_transitions(
        self, current_state: str, actor_role: str
    ) -> List[Dict[str, Any]]:
        """
        Get possible transitions from the current state for the given actor role.

        NOTE: Excludes IneffectiveEvent transitions as these are now handled
        explicitly through the contradiction analysis approach.

        Args:
            current_state: The current state in the legal process
            actor_role: The role of the actor performing the action

        Returns:
            List of dictionaries containing transition details including name,
            next_state, requirements, and other metadata
        """
        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (from:Stage {term: $state})-[:TRANSITIONS_TO]->(arc:Arc)-[:TRANSITIONS_TO]->(to:Stage)
                WHERE (arc.by = $role OR arc.by = 'Party' OR arc.by = 'Any')
                  AND NOT arc.term STARTS WITH $ineffective_prefix
                OPTIONAL MATCH (arc)-[:REQUIRES]->(req)
                WHERE NOT $interim_label IN labels(req)  // Skip interim nodes
                RETURN arc.term as name, arc.sense as details, arc.by as by, to.term as next_state,
                   collect({term: req.term, logic: req.logic}) as requirements
            """,
                state=current_state,
                role=actor_role,
                ineffective_prefix=LegalAnalyzerConfig.INEFFECTIVE_EVENT_PREFIX,
                interim_label=LegalAnalyzerConfig.INTERIM_STEP_LABEL,
            ).data()

        return results

    def _get_actor_role(
        self, event: LegalEvent, current_state: str, path_history: List[Dict[str, Any]]
    ) -> str:
        """
        Determine the legal role of an actor using LLM, considering legal state and path history.

        Args:
            event: Current legal event with actor information (Seller, Buyer, etc.)
            current_state: Current legal state in the reasoning path (NoLegalRelation, OfferPending, ContractExists, ModificationPending, etc.)
            path_history: List of past events in this reasoning path (LegalEvent)

        Returns:
            String representing the determined legal role ("Offeror", "Offeree", "Party", "Counterparty", "Third Party", etc.)
        """

        # Build context about pending offers from history
        existing_offers_context = ""
        existing_offers = []

        if path_history:
            for past_event in path_history:
                # Look for offer-related transitions that succeeded
                transition = past_event.get("transition", "")
                if (
                    "offer" in transition.lower() or "proposal" in transition.lower()
                ) and not transition.startswith("Failed"):
                    existing_offers.append(
                        {
                            "offeror": past_event.get("actor"),
                            "transition": transition,
                            "evidence": past_event.get("evidence", ""),
                        }
                    )

        if existing_offers:
            existing_offers_context = ""
            for i, offer in enumerate(existing_offers, 1):
                existing_offers_context += f"{i}. {offer['offeror']} made an offer ({offer['transition']}). Evidence from legal case: {offer['evidence']}\n"

        actor_role_determination_prompt = f"""
        You are a component in a legal analysis system. Your ONLY task is to assign a legal role to an actor based on the STRICT 'current_state' provided.

        **CRITICAL: DO NOT PREDICT THE OUTCOME OF THE CURRENT EVENT.** Do not assume the event will succeed, fail, or change the legal state. Your role assignment MUST be consistent with the 'current_state' provided as input. Base your decision on the rules below.

        ## Input Data
        - Current Actor: {event.actor}
        - Current Action: {event.action}
        - Current State: {current_state}
        - History of prior offers (chronological): {existing_offers_context}

        ## Role Assignment Rules (Apply these strictly)

        1.  If Current State is 'NoLegalRelation':
            - The actor PERFORMING the action (e.g., "sent telegram") is the 'Offeror'.
            - The actor RECEIVING the action is the 'Offeree'.

        2.  If Current State is 'OfferPending':
            - Identify the actor who made the MOST RECENT offer in the 'History'. This actor is the current 'Offeror'.
            - The other actor is the 'Offeree'.
            - NOTE: A counter-offer flips the roles, making the original Offeree the new Offeror.
            - Your final assigned role must be either 'Offeror' or 'Offeree'. The specific 'Current Action' does not change these roles within this state.

        3.  If Current State is 'ContractExists' or 'ModificationPending':
            - The actor has two roles simultaneously: a fixed role from contract formation and a dynamic role that depends on the nature of the current action.

            - **Step 1: Assign Fixed Role ('Offeror'/'Offeree')**
                - Identify the actor who made the LAST offer/counter-offer in the 'History' that led to the contract. This actor's fixed role is 'Offeror'.
                - The other actor's fixed role is 'Offeree'.
                - Determine which of these two fixed roles applies to the 'Current Actor'.

            - **Step 2: Assign Dynamic Role ('Party'/'Counterparty')**
                - To determine the dynamic role, compare the 'Current Actor' to the 'Actor of the immediately preceding event'.
                - If the actors are DIFFERENT, the 'Current Actor' is RESPONDING, and their dynamic role is 'Counterparty'.
                - If the actors are the SAME, or if there was no preceding event, the 'Current Actor' is INITIATING a new action, and their dynamic role is 'Party'.
        

        ## Your Task
        1. Identify the 'Current State'.
        2. Apply the rule for that state.
        3. If the state is 'OfferPending', 'ContractExists', or 'ModificationPending', use the 'History' to find the most recent offeror.
        4. Assign the correct role or roles to the 'Current Actor'.

        ## Output Requirements
        Provide your analysis and conclusion in JSON format with the fields "legal_role" and "reasoning". Keep the reasoning focused on which rule you applied.
        - For 'ContractExists' or 'ModificationPending' states, the "legal_role" field MUST be a JSON list containing BOTH the fixed role ("Offeror" or "Offeree") and the dynamic role ("Party" or "Counterparty"). For example: ["Offeror", "Party"].
        - For 'NoLegalRelation' or 'OfferPending' states, "legal_role" will be a single string.

        """

        # print(actor_role_determination_prompt)
        response = self._llm(actor_role_determination_prompt, output_schema=LegalRole)

        return response.legal_role, response.reasoning

    @retry_llm_call(
        max_retries=LegalAnalyzerConfig.MAX_RETRIES,
        base_delay=LegalAnalyzerConfig.BASE_DELAY,
        max_delay=LegalAnalyzerConfig.MAX_DELAY,
    )
    def _llm(self, prompt: str, output_schema: Optional[type] = None) -> Any:
        """
        LLM API call with structured output if specified.

        Args:
            prompt: The prompt to send to the LLM
            output_schema: The Pydantic model class to use for structured output

        Returns:
            Either a parsed Pydantic object (if output_schema provided) or raw text response
        """
        if output_schema:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": output_schema,
                    "temperature": LegalAnalyzerConfig.TEMPERATURE,
                },
            )
            if response.parsed is None:
                print(response)
                raise ValueError(f"No response from LLM for prompt: {prompt}")
            return response.parsed
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": LegalAnalyzerConfig.TEMPERATURE,
                },
            )
            if response.text is None:
                raise ValueError(f"No response from LLM for prompt: {prompt}")

            # Handle different response structures
            if hasattr(response, "text") and response.text:
                return response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        # Access text from parts array
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                return part.text
                    elif hasattr(candidate.content, "text") and candidate.content.text:
                        # Direct text access
                        return candidate.content.text

            # If we get here, no text was found
            raise ValueError(f"No text content found in response: {response}")

    @staticmethod
    def is_failed_transition(transition: str) -> bool:
        """Check if a transition represents any type of failed transition."""
        return transition in [
            LegalAnalyzerConfig.NO_TRANSITION,
            LegalAnalyzerConfig.FAILED_TRANSITION,
        ]

    def close(self) -> None:
        """Clean up database connection and release resources."""
        self.driver.close()

    def get_naive_llm_reply(self, prompt: str) -> str:
        """
        Get a naive LLM reply to a prompt.
        """

        NAIVE_PROMPT = """
PERSONA: You are an expert legal analyst, experienced with contract law cases. 

TASK: You will be given a legal case and are request to carry out a comprehensive legal analysis of the case, providing possible interpretations and the reasoning / justifications for each of the interpretations.

LEGAL CASE:
{prompt}

GUIDELINES:
- Focus on Argumentation: Look for legal, factual, or procedural reasons for your task.
- Ground Arguments in Facts: Base your reasoning strictly on the details provided in the context and legal rules.
- Take into account Ambiguity and possible interpretations
- Keep your reasoning short, concise, straightforward and to the point, while containing all the information needed to make a decision.

OUTPUT:
- Provide a comprehensive legal analysis of the case, providing possible interpretations and the reasoning / justifications for each of the interpretations.
"""

        response = self._llm(NAIVE_PROMPT, output_schema=LegalAnalysis)
        for i, (interpretation, justification) in enumerate(
            zip(response.interpretations, response.justifications)
        ):
            print(f"Interpretation {i}:")
            print(f"Interpretation: {interpretation}")
            print(f"Justification: {justification}\n")
            print("-" * 100)
        return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument(
        "--resume-event", type=int, default=None, help="Event number to resume from"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=f"{LegalAnalyzerConfig.DEFAULT_LOG_DIR}/legal_reasoning_log.txt",
        help="Path to log file (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--no-timestamps", action="store_true", help="Disable timestamps in log file"
    )
    args = parser.parse_args()

    LEGAL_NARRATIVE = """
    On July 1 Buyer sent the following telegram to Seller: "Have customers for salt and need
    carload immediately. Will you supply carload at $2.40 per cwt?"
    Seller received the telegram the same day. 
    On July 12 Seller sent Buyer the following telegram, which Buyer received the same day:
    "Accept your offer carload of salt, immediate shipment, terms cash on delivery."
    On July 13 Buyer sent by Air Mail its standard form "Purchase Order" to Seller. On the face of 
    the form Buyer had written that it accepted "Seller's offer of July 12" and had written 
    "One carload and $2.40 per cwt." in the appropriate spaces for quantity and price. Among 
    numerous printed provisions on the reverse of the form was the following: 
    "Unless otherwise stated on the face hereof, payment on all purchase orders shall not be due 
    until 30 days following delivery." There was no statement on the face of the form regarding 
    time of payment. 
    Later on July 13 another party offered to sell Buyer a carload of salt for $2.30 per cwt. 
    Buyer immediately wired Seller: "Ignore purchase order mailed earlier today; your offer of 
    July 12 rejected." 
    This telegram was received by Seller on the same day (July 13). 
    Seller received Buyer's purchase order in the mail the following day (July 14).
    Briefly analyze each of the items of correspondence in terms of its legal effect and indicate
    what the result will be in Seller's action against Buyer for breach of contract.
    """

    # Create logs directory if it doesn't exist
    os.makedirs(LegalAnalyzerConfig.DEFAULT_LOG_DIR, exist_ok=True)

    # Initialize logging
    with TeeOutput(log_file=args.log_file, include_timestamp=not args.no_timestamps) as logger:

        # Initialize analyzer
        analyzer = LegalAnalyzer(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD"),
            os.getenv("GEMINI_API_KEY"),
        )

        # for idx in range(6):
        #     print(f"Naive LLM Analysis  (iteration {idx+1}):")
        #     analyzer.get_naive_llm_reply(LEGAL_NARRATIVE)
        #     print("\n\n")

        try:
            # Load existing saved progress
            if args.resume:
                events, paths, event_num = LegalAnalyzer.load_progress(event_num=args.resume_event)
                reasoning_paths, events = analyzer.analyze(
                    events=events,
                    paths=paths,
                    start_event_num=event_num + 1,
                )
            else:
                # Default: Run new analysis
                reasoning_paths, events = analyzer.analyze(LEGAL_NARRATIVE)

            # Generate comprehensive summary table and statistics
            summary_stats = analyzer.generate_summary_table(
                events, reasoning_paths, LegalAnalyzerConfig.DEFAULT_SUMMARY_FILE
            )

        finally:
            analyzer.close()
