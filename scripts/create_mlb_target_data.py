import os
import json
from mlb_utils import get_player_line, get_team_line_attributes, match_in_candidate_innings
from mlb_utils import get_inning_side_entities
from mlb_utils import get_all_paragraph_plans
from mlb_utils import get_ents, get_ordinal_adjective_map, get_inning, sort_files_key, filter_summaries, \
    extract_entities, get_players_with_map
from mlb_utils import get_play_by_play_all_entities_inning, replace_carmona
from collections import OrderedDict
import logging
import argparse
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)
type_map = {"train": "train", "test": "test", "valid": "validation"}


def process(type, ordinal_adjective_map_file, output_folder):
    mlb_dataset = load_dataset("GEM/mlb_data_to_text")
    paragraph_plans_file = open(os.path.join(output_folder, type + ".pp"), mode="w", encoding="utf-8")
    macroplan_file = open(os.path.join(output_folder, type + ".macroplan"), mode="w", encoding="utf-8")
    summary_file = open(os.path.join(output_folder, type + ".su"), mode="w", encoding="utf-8")
    ordinal_adjective_map = get_ordinal_adjective_map(ordinal_adjective_map_file)
    seen_output = set()
    test_seen_output = set()
    if type == "train":
        #for filename in sorted_file_list:
        for candidate_type in ["valid", "test"]:
            print("test filename", candidate_type)
            data = mlb_dataset[type_map[candidate_type]]
            for entry_index, entry in enumerate(data):
                test_seen_output.add("_".join(entry["summary"][:50]))

    data = mlb_dataset[type_map[type]]
    for entry_index, entry in enumerate(data):
        logging.debug("instance %s", entry_index)
        if type == "train":
            if filter_summaries(entry, seen_output, test_seen_output):
                continue
        seen_output.add("_".join(entry["summary"][:50]))

        summary = entry["summary"]
        summ = " ".join(summary)
        if "Fausto Carmona" in summ:
            entry = replace_carmona(entry)

        all_ents, players, teams, cities = get_ents(entry)
        players_list, player_team_map = get_players_with_map(entry)
        segments = summ.split(" *NEWPARAGRAPH* ")
        if len(summary)//len(segments) < 15:  # such summaries seem to have incorrect
            # tokenization into paragraphs
            continue
        summary_segments = []
        names_map = {}
        candidate_innings = [
            len(entry["play_by_play"])]  # initialize with the last inning as it often occurs in the game
        descs = []
        entities_in_game = set()

        for segment in segments:
            _, sequential_entities = extract_entities(entry, segment.split(), all_ents, players, teams,
                                                         cities, players_list, names_map)
            entities_in_game.update([ent[0] for ent in sequential_entities if ent[1] is None])  # adding player entities

        logging.debug("entities_in_game %s", entities_in_game)
        for j, segment in enumerate(segments):
            logging.debug("segment %s", segment)
            prev_segment = [] if j == 0 else segments[j - 1].split()
            innings = get_inning(segment.split(), prev_segment, ordinal_adjective_map)
            ents, sequential_entities = extract_entities(entry, segment.split(), all_ents, players, teams, cities,
                                                         players_list, names_map)
            logging.debug("ents, sequential_entities  %s  %s", ents, sequential_entities)
            logging.debug("innings %s", innings)
            candidate_innings.extend([inn[0] for inn in innings])
            desc, inning_found = get_pbyp_desc(candidate_innings, entry, innings, sequential_entities)
            if len(" ".join(desc).split()) > 1200:  # include scoring runs
                desc, _ = get_pbyp_desc(candidate_innings, entry, innings, sequential_entities, apply_filter=True)
            if len(" ".join(desc).split()) > 1200:  # include scoring runs
                desc, _ = get_pbyp_desc(candidate_innings, entry, innings, sequential_entities, apply_filter=True, scoring=True)

            if not inning_found:
                desc = []
                if ents:
                    sequential_entities_upd = list(OrderedDict.fromkeys(sequential_entities))  # get non-repeating list
                    for entity in sequential_entities_upd:
                        if entity[1] is None:  # type is player
                            desc.append(get_player_line(entry["box_score"], entity[0]))
                        else:  # type is team
                            desc.append(get_team_line_attributes(entry, entity[0]))

            logging.debug("desc %s", desc)
            if j == len(segments) - 1 and (".. .." in segment or "..." in segment):  # ignore notes
                pass
            elif desc:
                descs.append(" ".join(desc))
                summary_segments.append(segment)
            elif j == 0:
                descs.append("<empty-segment>")  # ignore intermediate empty segments
                summary_segments.append(segment)
            logging.debug("=========================")
        summary_segments.append("<end-summary>")
        assert len(summary_segments) == len(descs) + 1
        augmented_paragraph_plans = get_all_paragraph_plans(entry, entry_index)
        prefix_tokens_ = ["<unk>", "<blank>", "<s>", "</s>", "<end-plan>", "<empty-segment>"]
        new_descs_list = prefix_tokens_ + augmented_paragraph_plans + descs
        new_descs_list = list(OrderedDict.fromkeys(new_descs_list))
        input_template = " ".join(new_descs_list[: len(prefix_tokens_)]) + " <segment> " + " <segment> ".join(
            new_descs_list[len(prefix_tokens_):])
        paragraph_plans_file.write(input_template)
        paragraph_plans_file.write("\n")
        chosen_paragraph_plans = [str(new_descs_list.index(descs[_paragraph])) for _paragraph in
                                  range(len(descs))]
        chosen_paragraph_plans += [str(new_descs_list.index("<end-plan>"))]
        macroplan_file.write(" ".join(chosen_paragraph_plans))
        macroplan_file.write("\n")
        summary_file.write("<segment> ")
        summary_file.write(" <segment> ".join(summary_segments))
        summary_file.write("\n")
        if entry_index % 50 == 0:
            print("entry_index", entry_index)
        #assert False

    paragraph_plans_file.close()
    macroplan_file.close()
    summary_file.close()


def get_pbyp_desc(candidate_innings, entry, innings, sequential_entities, apply_filter=False, scoring=False):
    desc = []
    inning_match = match_in_candidate_innings(entry, candidate_innings[::-1],
                                              [inn[0] for inn in innings],
                                              set([ent[0] for ent in sequential_entities
                                                   if ent[1] is None]))
    inning_found = False
    if len(innings) > 0 or inning_match != -1:
        p_by_p_desc = []
        total_entities_found = []
        total_entities = [ent[0] for ent in sequential_entities]
        innings_non_repeating = list(OrderedDict.fromkeys([x[0] for x in innings]))
        count_non_repeating_innings = len(innings_non_repeating) + (1 if inning_match != -1 else 0)
        for inning in innings_non_repeating:
            side, _ = get_inning_side_entities(entry, inning, total_entities)
            if side == "both":
                inning_found = True
                for each_side in ["top", "bottom"]:
                    run_pbyp(entry, inning, p_by_p_desc, each_side, total_entities_found,
                             total_entities, count_non_repeating_innings, apply_filter=apply_filter, scoring=scoring)
            elif side in ["top", "bottom"]:
                inning_found = True
                run_pbyp(entry, inning, p_by_p_desc, side, total_entities_found, total_entities,
                         count_non_repeating_innings, apply_filter=apply_filter, scoring=scoring)
        if inning_match != -1:
            logging.debug("inning_match %s", inning_match)
            side, _ = get_inning_side_entities(entry, inning_match, total_entities)
            if side in ["top", "bottom"]:  # ignore both side for inning_match
                inning_found = True
                run_pbyp(entry, inning_match, p_by_p_desc, side, total_entities_found, total_entities,
                         count_non_repeating_innings, apply_filter=apply_filter, scoring=scoring)

        total_entities_found.extend(
            [ent[0] for ent in sequential_entities if ent[1] is None])  # ensure all entities are accounted for;
        # some entities may be missed if side resolution is inconclusive
        total_entities_found = list(OrderedDict.fromkeys(total_entities_found))  # get non-repeating list

        desc.append(get_team_line_attributes(entry, entry["home_line"]["team_name"]))
        desc.append(get_team_line_attributes(entry, entry["vis_line"]["team_name"]))
        desc.extend([get_player_line(entry["box_score"], player_name) for player_name in total_entities_found])
        desc.extend(p_by_p_desc)
    return desc, inning_found


def run_pbyp(entry, inning, p_by_p_desc, side, total_entities_found, total_entities_in_segment,
             count_non_repeating_innings, apply_filter=False, scoring = False):
    play_by_play_desc, entities_found = get_play_by_play_all_entities_inning(entry,
                                                                             entry[
                                                                                 "home_line"][
                                                                                 "team_name"],
                                                                             entry[
                                                                                 "vis_line"][
                                                                                 "team_name"],
                                                                             inning,
                                                                             side,
                                                                             total_entities_in_segment,
                                                                             count_non_repeating_innings,
                                                                             apply_filter=apply_filter,
                                                                             scoring=scoring)
    total_entities_found.extend(entities_found)
    p_by_p_desc.extend(play_by_play_desc)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting entities from summary file')
    parser.add_argument('-ordinal_adjective_map_file', type=str,
                        help='path of ordinal_adjective_map_file', default=None)
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    args = parser.parse_args()
    process(args.dataset_type, args.ordinal_adjective_map_file, args.output_folder)
