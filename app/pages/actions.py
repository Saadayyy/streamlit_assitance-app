from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd


class ActionDataSummary(Action):
    def name(self) -> Text:
        return "action_data_summary"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        column_name = next(tracker.get_latest_entity_values("column_name"), None)
        if column_name:
            try:
                dataset = pd.read_csv("data/dataset.csv")  # Path to your dataset
                if column_name in dataset.columns:
                    avg = dataset[column_name].mean()
                    dispatcher.utter_message(
                        text=f"The average of '{column_name}' is {avg:.2f}."
                    )
                else:
                    dispatcher.utter_message(
                        text=f"Column '{column_name}' not found in the dataset."
                    )
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing data: {str(e)}")
        else:
            dispatcher.utter_message(
                text="Please specify the column you'd like to analyze."
            )
        return []


class ActionFeatureGuidance(Action):
    def name(self) -> Text:
        return "action_feature_guidance"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        feature_name = next(tracker.get_latest_entity_values("feature_name"), None)
        feature_guide = {
            "remove outliers": "Go to the 'Outlier Handling' section and check the 'Remove Outliers' option.",
            "visualize data": "Navigate to 'Data Visualizations' and select the desired chart type.",
        }
        response = feature_guide.get(
            feature_name.lower(), "Sorry, I couldn't find guidance for that feature."
        )
        dispatcher.utter_message(text=response)
        return []
