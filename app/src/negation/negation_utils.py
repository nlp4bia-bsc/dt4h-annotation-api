"""
negation.py

This module handles negation and uncertainty detection on NER results.
It uses a negation/uncertainty tagger model to identify NEG/NSCO/UNC/USCO entities
and enriches NER entities with is_negated and is_uncertain attributes.

Author: Jan Rodríguez Miret
"""

def add_negation_uncertainty_attributes(nerl_results: list[list[dict]], negation_entities_list: list[list[dict]]) -> list[list[dict]]:
    """
    Add is_negated and is_uncertain attributes to NER entities based on overlap with negation/uncertainty scope entities.
    Also includes negation/uncertainty scores from the matching scopes.
    Calls the negation tagger internally to detect negation/uncertainty scopes.
    Filters out NEG, NSCO, UNC, USCO entities from the results.
    
    Args:
        nerl_results (list): List of NER entities for each text (output from ner_inference with combined=True)
        texts (list): List of original text strings to process with negation tagger
        agg_strat (str): Aggregation strategy for the negation tagger model
    
    Returns:
        list: List of NER entities with added is_negated, is_uncertain, negation_score, and uncertainty_score attributes
    """
    # Run negation tagger inference to get negation/uncertainty entities
    results_with_attributes = []
    
    for nerL_entities_doc, negation_entities_doc in zip(nerl_results, negation_entities_list):
        
        # Separate negation/uncertainty scopes and triggers
        neg_scopes = [e for e in negation_entities_doc if e['ner_class'] == 'NSCO']
        unc_scopes = [e for e in negation_entities_doc if e['ner_class'] == 'USCO']
        
        # Filter NER entities: exclude NEG, NSCO, UNC, USCO entities
        filtered_ner_entities = [
            e for e in nerL_entities_doc 
            if e['ner_class'] not in ['NEG', 'NSCO', 'UNC', 'USCO'] # though idk why this would be called if the neg ner is only called above
        ]
        
        # Add negation/uncertainty attributes to each entity
        for entity in filtered_ner_entities:
            is_negated, negation_score = _find_property(entity, neg_scopes)
            is_uncertain, uncertainty_score = _find_property(entity, unc_scopes)
            
            entity['is_negated'] = is_negated
            entity['negation_score'] = negation_score
            entity['is_uncertain'] = is_uncertain
            entity['uncertainty_score'] = uncertainty_score
        
        results_with_attributes.append(filtered_ner_entities)
    
    return results_with_attributes

def _find_property(entity: dict, prop_scopes: list[dict]) -> tuple[int, float | None]:
    # Find overlapping negation scopes and get their scores
    overlapping_scopes = [scope for scope in prop_scopes if _entity_in_scope(entity, scope)] # for some reason, there can be more than one negation per entity
    # Use the highest score if multiple scopes overlap
    return len(overlapping_scopes), max([scope['ner_score'] for scope in overlapping_scopes]) if overlapping_scopes else None

def _entity_in_scope(entity: dict, scope_ent: dict) -> bool:
    """
    Check if two entities overlap based on their start and end positions.
    Returns True if entity1 is within or overlaps with entity2's scope.
    
    Args:
        entity (dict): Entity (clinical) with 'start' and 'end' keys
        scope_ent (dict): Entity (NSCO/USCO) with 'start' and 'end' keys
    
    Returns:
        bool: True if entities overlap, False otherwise
    """
    return (
        # entity starts within scope
        (scope_ent['start'] <= entity['start'] < scope_ent['end']) or
        # entity ends within scope
        (scope_ent['start'] < entity['end'] <= scope_ent['end'])
        # TODO: Uncomment for considering negated the case where scope is "nested-smaller" in 
        # brat-peek, i.e., clin_ent completely contains NSCO/USCO ([cancer not detected])
        # or (entity1['start'] <= entity2['start'] and entity1['end'] >= entity2['end'])
    )
