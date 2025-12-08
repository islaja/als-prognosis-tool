import csv
from typing import List, Dict

def read_participant_inputs(path: str) -> List[Dict]:
    """Read a Participant_Inputs_File.csv with columns:
    ParticipantID,ParticipantVisit,Path

    Returns a list of dicts.
    """
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # ensure keys are exactly ParticipantID, ParticipantVisit, Path
            normalized = {
                'ParticipantID': r.get('ParticipantID') or r.get('participantid') or r.get('participant_id'),
                'ParticipantVisit': r.get('ParticipantVisit') or r.get('participantvisit') or r.get('visit'),
                'Path': r.get('Path') or r.get('path') or r.get('PathToT1w') or r.get('PathToFeatureCSV')
            }
            rows.append(normalized)
    return rows
