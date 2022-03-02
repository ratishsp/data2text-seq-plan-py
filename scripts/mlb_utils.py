import re
from collections import OrderedDict
from nltk.corpus import stopwords
import json

team_line1 = "%s <TEAM> %s <CITY> %s <TEAM-RESULT> %s <TEAM_RUNS> %d <TEAM_HITS> %d <TEAM_ERRORS> %d"

pbyp_verbalization_map = {"o": "<PBYP-OUTS>", "b": "<PBYP-BALLS>", "s": "<PBYP-STRIKES>", "b1": "<PBYP-B1>",
                     "b2": "<PBYP-B2>", "b3": "<PBYP-B3>", "batter": "<PBYP-BATTER>", "pitcher": "<PBYP-PITCHER>",
                     "scorers": "<PBYP-SCORERS>", "event": "<PBYP-EVENT>", "event2": "<PBYP-EVENT2>",
                     "fielder_error": "<PBYP-FIELDER-ERROR>", "runs": "<PBYP-RUNS>", "rbi": "<PBYP-RBI>",
                     "error_runs": "<PBYP-ERROR-RUNS>", "top": "<TOP>", "bottom": "<BOTTOM>"}
pitcher_verbalization_map = {"p_bb": "<PITCH-BASE-ON-BALLS>", "p_er": "<EARNED-RUN>", "p_era": "<EARNED-RUN-AVG>",
                     "p_h": "<PITCH-HITS>", "p_hr": "<PITCH-HOME-RUN>", "p_l": "<PITCH-LOSS>",
                     "p_loss": "<PITCH-LOSING-PITCHER>", "p_s": "<PITCH-STRIKES-THROWN>",
                     "p_np": "<PITCH-COUNT>", "p_r": "<PITCH-RUNS>", "p_save": "<PITCH-SAVING-PITCHER>",
                     "p_so": "<PITCH-STRIKE-OUT>", "p_bf": "<PITCH-BATTERS-FACED>", "p_bs": "<PITCH-BLOWN-SAVE>",
                     "p_sv": "<PITCH-SAVE>", "p_w": "<PITCH-WIN>", "p_ip1": "<INNINGS-PITCHED-1>",
                     "p_ip2": "<INNINGS-PITCHED-2>", "p_win": "<PITCH-WINNING-PITCHER>", "p_out": "<PITCH-OUT>"}
batter_verbalization_map = {"h": "<HITS>", "r": "<RUNS>", "hr": "<HOME-RUN>", "ab": "<ATBAT>", "avg": "<AVG>",
                     "rbi": "<RBI>", "cs": "<CAUGHT-STEAL>", "hbp": "<HIT-BY-PITCH>", "a": "<ASSIST>",
                     "bb": "<BASE-ON-BALL>", "e": "<ERROR>", "obp": "<ON-BASE-PCT>", "po": "<PUTOUT>",
                     "pos": "<POS>", "sb": "<STOLEN-BASE>", "sf": "<SAC-FLY>", "slg": "<SLUG>",
                     "so": "<STRIKEOUT>"
                             }
player_verbalization_map = dict(pitcher_verbalization_map, **batter_verbalization_map)
team_verbalization_map = {"team_errors": "<TEAM_ERRORS>", "team_hits": "<TEAM_HITS>", "team_runs": "<TEAM_RUNS>"}
HIGH_NUMBER = -100


def get_team_line_attributes(entry, name):
    """

    :param entry:
    :param name:
    :return:
    """
    if name == entry["home_line"]["team_name"]:
        line = entry["home_line"]
        type = "home"
    elif name == entry["vis_line"]["team_name"]:
        line = entry["vis_line"]
        type = "vis"
    else:
        assert False

    city = line["team_city"]
    name = line["team_name"]
    result = line["result"]
    updated_type = "<"+type.upper()+">"
    team_tup = (updated_type, name, city, result)
    team_line = "%s <TEAM> %s <CITY> %s <TEAM-RESULT> %s"
    sentence1 = team_line %(team_tup)
    other_attributes = []
    attributes = ["team_runs", "team_hits", "team_errors"]
    for attrib in attributes:
        template_string = " ".join([team_verbalization_map[attrib], "%s"])
        other_attributes.append(template_string % line[attrib])
    other_attributes = " ".join(other_attributes)
    team_info = sentence1
    if len(other_attributes) > 0:
        team_info = " ".join([sentence1, other_attributes])
    return team_info


def get_player_line(bs, input_player_name):
    """
    :param bs:
    :param input_player_name:
    :return:
    """
    player_line = "<PLAYER> %s <TEAM> %s <POS> %s"
    player_found = False
    player_info = ""
    for bs_entry in bs:
        if bs_entry["full_name"] == input_player_name:
            updated_name = bs_entry["full_name"]
            if "  " in bs_entry["full_name"]:
                updated_name = bs_entry["full_name"].replace("  ", " ")
            elif bs_entry["full_name"].endswith(" "):
                updated_name = bs_entry["full_name"].strip()
            pos = bs_entry["pos"]
            if not pos.strip():
                pos = "N/A"
            player_tup = (tokenize_initials(updated_name), bs_entry["team"], pos)
            player_basic_info = player_line %(player_tup)
            other_attributes = []
            for attrib in ["r", "h", "hr", "rbi", "e", "ab", "avg", "cs", "hbp", "bb", "sb", "sf", "so", "a", "po",
                           "p_ip1", "p_ip2", "p_w", "p_l", "p_h", "p_r", "p_er", "p_bb", "p_so", "p_hr", "p_np", "p_s",
                           "p_era", "p_win", "p_loss", "p_save", "p_sv", "p_bf", "p_out", "p_bs"]:
                if bs_entry[attrib] == "N/A":
                    continue
                if attrib in ['sb', 'sf', 'e', 'po', 'a', 'cs', 'hbp', 'hr', 'so', 'bb', "p_hr", "p_sv",
                              "p_bs"] and int(bs_entry[attrib]) == 0:
                    continue
                if attrib in ['avg']  and bs_entry[attrib] == ".000":
                    continue
                template_string = " ".join([player_verbalization_map[attrib], "%s"])
                other_attributes.append(template_string %(bs_entry[attrib]))
            if other_attributes:
                player_other_attributes = " ".join(other_attributes)
                player_info = " ".join([player_basic_info, player_other_attributes])
            else:
                player_info = player_basic_info
            player_found = True
    assert player_found
    return player_info


def get_play_by_play_all_entities_inning(entry, home, away, inning, side, total_entities_in_segment=None,
                                         count_non_repeating_innings=0, apply_filter=False, scoring=False):
    """
    Method to get play by play for all entities given an inning and side
    :param entry:
    :param home:
    :param away:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    play_by_play_desc = []
    if inning > len(plays):
        return play_by_play_desc, None

    play_index = 1
    inning_plays = plays[inning - 1][side]
    if side == "top":
        batting_team = away
        inning_runs = entry["vis_line"]["innings"][inning - 1]["runs"]
    else:
        batting_team = home
        inning_runs = entry["home_line"]["innings"][inning - 1]["runs"]
    assert entry["home_line"]["innings"][inning - 1]["inn"] == str(inning)

    inning_runs_vb = "<INNING-RUNS> %s %s" % (batting_team, inning_runs)
    prev_scores = get_prev_scores(entry, inning, side)
    prev_scores_vb = "<PREV-SCORES> %s %d %s %d" % (
        home, prev_scores[0], away, prev_scores[1])
    inning_info_added = False
    total_entities_found = []
    for inning_play in inning_plays:
        entities_found = []
        other_attrib_desc = get_play_by_play_desc(entities_found, home,
                                                                away, inning, inning_play, play_index, side)
        play_index += 1
        if apply_filter:
            if inning_play["runs"] == "0":
                if scoring or (len(set(entities_found).intersection(set(total_entities_in_segment))) == 0):
                    continue
        if True:
            if not inning_info_added:
                other_attrib_desc.insert(1, prev_scores_vb)
                other_attrib_desc.insert(2, inning_runs_vb)
                inning_info_added = True
            other_attrib_desc = " ".join(other_attrib_desc)
            play_by_play_desc.append(other_attrib_desc)
        total_entities_found.extend(entities_found)
    return play_by_play_desc, total_entities_found


def get_play_by_play_desc(entities_found, home, away, inning, inning_play, play_index,
                          top_bottom):
    inning_line = " ".join(["<INNING> %d", pbyp_verbalization_map[top_bottom], "<BATTING> %s <PITCHING> %s"])
    if top_bottom == "top":
        inning_attrib = (inning, away, home)
    else:
        inning_attrib = (inning, home, away)
    inning_desc = inning_line % (inning_attrib)
    other_attrib_desc = [inning_desc]
    play_vb= "<PLAY> %d" % play_index
    other_attrib_desc.append(play_vb)
    other_attrib_desc.extend(get_runs_desc(inning_play))
    other_attrib_desc.extend(get_obs_desc(inning_play))
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play and inning_play[attrib] != "N/A":
            entities_found.append(inning_play[attrib])
            get_name_desc(attrib, inning_play, other_attrib_desc)
    for attrib in ["scorers", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                entities_found.append(baserunner_instance)
                get_name_desc_entity(attrib, baserunner_instance, other_attrib_desc)
    get_attrib_value_desc("event", inning_play, other_attrib_desc)
    get_attrib_value_desc("event2", inning_play, other_attrib_desc)
    get_team_scores_desc(away, home, inning_play, other_attrib_desc)
    return other_attrib_desc


def get_prev_scores(entry, inning, top_bottom):
    home_line = entry["home_line"]
    vis_line = entry["vis_line"]
    prev_scores_map = {}
    for side, line in zip(["bottom", "top"],[home_line, vis_line]):
        prev_scores = [0]
        innings = line["innings"]
        for inning_entry in innings:
            if inning_entry["runs"] != "x":
                prev_scores.append(prev_scores[-1] + int(inning_entry["runs"]))
        prev_scores_map[side] = prev_scores
    if top_bottom == "top":  # report previous inning scores
        return prev_scores_map["bottom"][inning - 1], prev_scores_map["top"][inning - 1]
    else:  # report vis team (top) same inning scores and home team (bottom) prev inning scores
        return prev_scores_map["bottom"][inning - 1], prev_scores_map["top"][inning]


def get_inning_side_entities(entry, inning, entities):
    """
    Method to get side of the inning described in the summary
    :param entry:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    if inning > len(plays):
        return None, None

    entities_so_far_side = []
    total_non_batting_entities = []
    for top_bottom in ["top", "bottom"]:
        non_batting_entities = []
        inning_plays = plays[inning - 1][top_bottom]
        all_entities_found = set()
        for inning_play in inning_plays:
            entities_found, non_batting = get_entities_in_play(entities, inning_play)
            non_batting_entities.append(non_batting)
            all_entities_found.update(entities_found)
        entities_so_far_side.append(all_entities_found)
        total_non_batting_entities.append(non_batting_entities)

    if not entities_so_far_side[0] and not entities_so_far_side[1]:  # no entities;
        return None, None
    if len(entities_so_far_side[0]) > len(entities_so_far_side[1]):
        return "top", entities_so_far_side[0]
    elif len(entities_so_far_side[0]) < len(entities_so_far_side[1]):
        return "bottom", entities_so_far_side[1]
    else:
        if any(total_non_batting_entities[0]):
            return "top", entities_so_far_side[0]
        elif any(total_non_batting_entities[1]):
            return "bottom", entities_so_far_side[1]
        else:
            return "both", None


def match_in_candidate_innings(entry, innings, summary_innings, entities):
    """
    :param entry:
    :param innings: innings to be searched in
    :param summary_innings: innings mentioned in the summary segment
    :param entities: total entities in the segment
    :return:
    """
    entities_in_summary_inning = set()
    for summary_inning in summary_innings:
        intersection = get_matching_entities_in_inning(entry, summary_inning, entities)
        entities_in_summary_inning.update(intersection)
    entities_not_found = entities.difference(entities_in_summary_inning)
    matched_inning = -1
    if len(entities_not_found) > 1:
        remaining_inings = set(innings).difference(set(summary_innings))
        orderered_remaining_innings = [inning for inning in innings if inning in remaining_inings]
        matched_inning = get_inning_all_entities_set_intersection(entry, orderered_remaining_innings, entities_not_found)
    return matched_inning


def get_entities_in_play(entities, inning_play):
    non_batting = False
    entities_found = set()
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play and inning_play[attrib] in entities:
            entities_found.add(inning_play[attrib])
            if attrib in ["fielder_error", "pitcher"]:
                non_batting = True
    for attrib in ["scorers", "b1", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                if baserunner_instance in entities:
                    entities_found.add(baserunner_instance)
    return entities_found, non_batting


def get_matching_entities_in_inning(entry, inning, entities):
    """
    Method to get matching entities in an inning with the summary
    :param entry:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    entities_in_inning = set()
    for top_bottom in ["top", "bottom"]:
        if inning <= len(plays):  # inning may be of a previous match like "He got the victory Friday when he got David Ortiz to hit into an inning-ending double play in the 11th inning"
            inning_plays = plays[inning - 1][top_bottom]
            for inning_play in inning_plays:
                for attrib in ["batter", "pitcher", "fielder_error"]:
                    if attrib in inning_play:
                        entities_in_inning.add(inning_play[attrib])
                for attrib in ["scorers", "b1", "b2", "b3"]:
                    if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
                        for baserunner_instance in inning_play[attrib]:
                            entities_in_inning.add(baserunner_instance)
    intersection = entities_in_inning.intersection(entities)
    return intersection


def get_inning_all_entities_set_intersection(entry, innings, entities):
    """
    Method to get inning
    :param entry:
    :param innings:
    :param entities:
    :return:
    """
    max_intersection = 1
    matched_inning = -1
    for inning in innings:
        intersection = get_matching_entities_in_inning(entry, inning, entities)
        if max_intersection < len(intersection):
            max_intersection = len(intersection)
            matched_inning = inning

    return matched_inning


def get_team_scores_desc(away, home, inning_play, obs_desc):
    if "home_team_runs" in inning_play and "away_team_runs" in inning_play and inning_play["home_team_runs"] != "N/A" \
            and inning_play["away_team_runs"] != "N/A":
        desc = "<TEAM-SCORES> %s %d %s %d" % (
            home, int(inning_play["home_team_runs"]), away, int(inning_play["away_team_runs"]))
        obs_desc.append(desc)


def get_attrib_value_desc(attrib, inning_play, obs_desc):
    if attrib in inning_play and inning_play[attrib] != "N/A":
        desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
        obs_desc.append(desc % (inning_play[attrib]))


def get_name_desc(attrib, inning_play, obs_desc):
    if attrib in inning_play:
        desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
        name = inning_play[attrib]
        if "  " in name:
            name = name.replace("  ", " ")
        elif name.endswith(" "):
            name = name.strip()
        attrib_value = tokenize_initials(name)
        obs_desc.append(desc % (attrib_value))


def get_name_desc_entity(attrib, entity_name, obs_desc):
    desc = " ".join([pbyp_verbalization_map[attrib], "%s"])
    name = entity_name
    if "  " in entity_name:
        name = entity_name.replace("  ", " ")
    elif entity_name.endswith(" "):
        name = entity_name.strip()
    attrib_value = tokenize_initials(name)
    obs_desc.append(desc % (attrib_value))


def get_runs_desc(inning_play):
    obs_desc = []
    for attrib in ["runs", "rbi", "error_runs"]:
        if attrib in inning_play and inning_play[attrib] != "N/A" and int(inning_play[attrib]) > 0:
            desc = " ".join([pbyp_verbalization_map[attrib], "%d"])
            obs_desc.append(desc % (int(inning_play[attrib])))
    return obs_desc


def get_obs_desc(inning_play):
    obs_desc = []
    for attrib in ["o", "b", "s"]:
        if attrib in inning_play:
            desc = " ".join([pbyp_verbalization_map[attrib], "%d"])
            obs_desc.append(desc % (int(inning_play[attrib])))
    return obs_desc


def tokenize_initials(value):
    attrib_value = re.sub(r"(\w)\.(\w)\.", r"\g<1>. \g<2>.", value)
    return attrib_value


def get_all_paragraph_plans(entry, entry_index):
    output = []
    if entry["home_line"]["result"] == "win":
        win_team_name = entry["home_name"]
        lose_team_name = entry["vis_name"]
    else:
        win_team_name = entry["vis_name"]
        lose_team_name = entry["home_name"]
    box_score = entry["box_score"]
    top_home_players = get_players(entry["box_score"], entry["home_name"])
    top_vis_players = get_players(entry["box_score"], entry["vis_name"])
    total_players = top_home_players + top_vis_players
    # teams
    output.append(get_team_line_attributes(entry, win_team_name))
    output.append(get_team_line_attributes(entry, lose_team_name))
    # both teams together
    output.append(" ".join(
        [get_team_line_attributes(entry, win_team_name),
         get_team_line_attributes(entry, lose_team_name)]))
    # opening statement
    for player_index, player in enumerate(total_players):
        output.append(" ".join(
            [get_player_line(box_score, player),
             get_team_line_attributes(entry, win_team_name),
             get_team_line_attributes(entry, lose_team_name)]))
    # each player
    for player_index, player in enumerate(total_players):
        output.append(get_player_line(box_score, player))
    # team and player
    """
    for team, players in zip([entry["home_name"], entry["vis_name"]], [top_home_players, top_vis_players]):
        for player in players:
            desc = " ".join(
                [get_team_line_attributes(entry, team),
                 get_player_line(box_score, player)])
            output.append(desc)
    """
    # pair of players in the same team
    """
    for player_seq in [top_home_players, top_vis_players]:
        for player_index, player in enumerate(player_seq):
            player_line = get_player_line(box_score, player)
            for player_2 in player_seq[player_index + 1:]:
                player_line_2 = get_player_line(box_score, player_2)
                output.append(" ".join([player_line, player_line_2]))
    """
    for inning in range(1, len(entry['home_line']['innings']) + 1):
        if entry["vis_line"]["innings"][inning - 1]["runs"] == 'x':
            # print("x midway in inning", str(entry_index))
            break
        for side in ["top", "bottom"]:
            pbyp_desc, entities_found = get_play_by_play_all_entities_inning(entry, entry["home_line"]["team_name"],
                                                                             entry["vis_line"]["team_name"], inning,
                                                                             side)
            entities_found = list(OrderedDict.fromkeys(entities_found))
            desc = []
            desc.append(get_team_line_attributes(entry, entry["home_line"]["team_name"]))
            desc.append(get_team_line_attributes(entry, entry["vis_line"]["team_name"]))
            desc.extend(
                [get_player_line(entry["box_score"], player_name) for player_name in entities_found])
            desc.extend(pbyp_desc)
            if pbyp_desc:
                output.append(" ".join(desc))
    return output


def get_players(bs, team):
    player_lists = []
    for key, bs_entry in enumerate(bs):
        if bs_entry["pos"] == "N/A":
            continue
        player_lists.append((key, get_attrib_value(bs_entry, "r"), get_attrib_value(bs_entry, "rbi"), get_attrib_value(bs_entry, "p_ip1")))
    player_lists.sort(key=lambda x: (-int(x[1]), -int(x[2]), -int(x[3])))
    players = []
    for (pid, _, _, _) in player_lists:
        if bs[pid]["team"]  == team:
            players.append(bs[pid]["full_name"])
    return players


def get_attrib_value(bs_entry, attrib):
    return bs_entry[attrib] if bs_entry[attrib] != "N/A" else HIGH_NUMBER


def get_play_by_play_all_entities_inning_gen(entry, home, away, inning, entities, side):
    """
    Method to get play by play for all entities given an inning and side
    :param entry:
    :param home:
    :param away:
    :param inning:
    :param entities:
    :return:
    """
    plays = entry["play_by_play"]
    play_by_play_desc = []
    if str(inning) not in plays:
        return play_by_play_desc, None

    play_index = 1
    inning_plays = plays[str(inning)][side]
    entities_found = []
    for inning_play in inning_plays:
        entity_found, other_attrib_desc = get_play_by_play_desc_gen(entities_found, entities, home,
                                                                away, inning, inning_play, play_index, side)
        other_attrib_desc = " ".join(other_attrib_desc)
        play_index += 1
        if entity_found:
            play_by_play_desc.append(other_attrib_desc)
    return play_by_play_desc, entities_found


def get_play_by_play_desc_gen(entities_found, entities_so_far, home, away, inning, inning_play, play_index,
                          top_bottom):
    entity_found = False
    inning_line = " ".join(["<INNING> %d", pbyp_verbalization_map[top_bottom], "<BATTING> %s <PITCHING> %s <PLAY> %d"])
    if top_bottom == "top":
        inning_attrib = (inning, away, home, play_index)
    else:
        inning_attrib = (inning, home, away, play_index)
    inning_desc = inning_line % (inning_attrib)
    other_attrib_desc = [inning_desc]
    other_attrib_desc.extend(get_runs_desc(inning_play))
    other_attrib_desc.extend(get_obs_desc(inning_play))
    for attrib in ["batter", "pitcher", "fielder_error"]:
        if attrib in inning_play and inning_play[attrib] in entities_so_far:
            entity_found = True
            entities_found.append(inning_play[attrib])
            get_name_desc(attrib, inning_play, other_attrib_desc)
    for attrib in ["scorers", "b2", "b3"]:
        if attrib in inning_play and len(inning_play[attrib]) > 0 and inning_play[attrib][0] != "N/A":
            for baserunner_instance in inning_play[attrib]:
                if baserunner_instance in entities_so_far:
                    entity_found = True
                    entities_found.append(baserunner_instance)
                    get_name_desc_entity(attrib, baserunner_instance, other_attrib_desc)
    get_attrib_value_desc("event", inning_play, other_attrib_desc)
    get_attrib_value_desc("event2", inning_play, other_attrib_desc)
    get_team_scores_desc(away, home, inning_play, other_attrib_desc)
    return entity_found, other_attrib_desc


def get_team_information(thing, home):
    teams = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"

    teams.add(thing[team_type + "name"])
    teams.add(" ".join([thing[team_type + "city"], thing[team_type + "name"]]))

    alternate_names = {"D-backs": "Diamondbacks", "Diamondbacks": "D-backs", "Athletics": "A 's"}
    for key in alternate_names:
        if thing[team_type + "name"] == key:
            teams.add(" ".join([thing[team_type + "city"], alternate_names[key]]))
            teams.add(alternate_names[key])
    return teams


def get_city_information(thing, home):
    cities = set()
    if home:
        team_type = "home_"
    else:
        team_type = "vis_"
    cities.add(thing[team_type + "city"])

    alternate_names = {"Chi Cubs": ["Chicago"], "LA Angels": ["Los Angeles", "LA"], "LA Dodgers": ["Los Angeles", "LA"],
                       "NY Yankees": ["New York", "NY"], "NY Mets": ["New York", "NY"], "Chi White Sox": ["Chicago"]}
    for key in alternate_names:
        if thing[team_type + "city"] == key:
            for val in alternate_names[key]:
                cities.add(val)
    return cities


def get_ents(thing):
    players = set()
    teams = set()
    cities = set()
    teams.update(get_team_information(thing, home=False))
    teams.update(get_team_information(thing, home=True))
    cities.update(get_city_information(thing, home=False))
    cities.update(get_city_information(thing, home=True))
    players.update([x["full_name"] for x in thing["box_score"]])
    players.update([x["last_name"] for x in thing["box_score"]])
    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    all_ents = players | teams | cities
    return all_ents, players, teams, cities


def get_team_idx(thing, entname):
    home_teams = set()
    home_cities = set()
    vis_teams = set()
    vis_cities = set()
    vis_teams.update(get_team_information(thing, home=False))
    home_teams.update(get_team_information(thing, home=True))
    vis_cities.update(get_city_information(thing, home=False))
    home_cities.update(get_city_information(thing, home=True))

    for entset in [home_teams, home_cities, vis_teams, vis_cities]:
        for k in list(entset):
            pieces = k.split()
            for piece_index in range(len(pieces)):
                entset.add(" ".join(pieces[:piece_index]))
    if entname in home_teams or entname in home_cities:
        team_name = (thing["home_name"], "home")
    elif entname in vis_teams or entname in vis_cities:
        team_name = (thing["vis_name"], "vis")
    else:
        assert False

    return team_name


def get_player_idx(players, entname, names_map, prev_word, prev_second_word):
    keys = []
    matched_player_name = None
    for index, v in enumerate(players):
        if entname == v[0] or entname + " Jr." == v[0]:
            names_map[v[1]] = v[0]
            matched_player_name = v[0]  # (full name, last name, first name)
    if len(keys) == 0:
        for index, v in enumerate(players):  # handling special cases
            if prev_second_word + prev_word == v[2] and entname == v[1]:  # handling tokenization of C.J as C. J.
                matched_player_name = v[0]
                names_map[v[1]] = v[0]
    if len(keys) == 0:
        if entname in names_map:
            matched_player_name = names_map[entname]
        elif entname + " Jr." in names_map:
            matched_player_name = names_map[entname + " Jr."]
        else:
            for index, v in enumerate(players):
                if entname == v[1]:  # matching second name
                    matched_player_name = v[0]
                elif entname + " Jr." == v[1]:
                    matched_player_name = v[0]
        if len(keys) > 1:
            print("prev_word", prev_word)
            print("more than one match", entname + ":" + str(bs["full_name"].values()))
            matched_player_name = None
    return matched_player_name


def get_ordinal_adjective_map(ordinal_adjective_map_file_name):
    ordinal_adjective_map_file = open(ordinal_adjective_map_file_name, mode="r", encoding="utf-8")
    ordinal_adjective_map_lines = ordinal_adjective_map_file.readlines()
    ordinal_adjective_map_lines = [line.strip() for line in ordinal_adjective_map_lines]
    ordinal_adjective_map = {}
    for line in ordinal_adjective_map_lines:
        ordinal_adjective_map[line.split("\t")[0]] = line.split("\t")[1]
    return ordinal_adjective_map


def get_inning(sent, prev_sent_context, ordinal_adjective_map):
    inning_identifier = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                         "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th"}
    inning_identifier_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                             "eighth": 8, "ninth": 9, "tenth": 10, "7th": 7, "8th": 8, "9th": 9, "10th": 10, "11th": 11,
                             "12th": 12, "13th": 13, "14th": 14, "15th": 15}
    stops = stopwords.words('english')
    innings = []
    upd_sent = " ".join(sent)
    upd_sent = upd_sent.replace("-", " ").split()  # handles cases such as these: pitched out of a second-inning jam
    intersected = set(upd_sent).intersection(inning_identifier)
    if len(intersected) > 0:
        # candidate present
        for i in range(len(sent)):
            if sent[i] in inning_identifier and i+1 < len(sent) and sent[i+1] in ["inning", "innings"]:
                innings.append((inning_identifier_map[sent[i]], i))
            elif "-" in sent[i]  and sent[i].split("-")[0] in inning_identifier and sent[i].split("-")[1]  == "inning":
                innings.append((inning_identifier_map[sent[i].split("-")[0]], i))
            elif (" ".join(sent[:i]).endswith("in the") or " ".join(sent[:i]).endswith("in the top of the") or " ".join(
                    sent[:i]).endswith("in the bottom of the")) and sent[i] in inning_identifier and (
                    (i + 1 < len(sent) and (sent[i + 1] in [".", ","] or sent[i + 1] in stops)) or i + 1 == len(sent)):
                innings.append((inning_identifier_map[sent[i]], i))
            elif sent[i] in inning_identifier and ((i+1 < len(sent) and (sent[i+1] in [".", ","] or sent[i+1] in stops)) or i+1 == len(sent)):
                # i+1 == len(sent) handles the case such as "Kapler also doubled in a run in the first "; no full stop at the end
                expanded_context = prev_sent_context + sent[:i+1]
                expanded_context = " ".join(expanded_context)
                assert expanded_context in ordinal_adjective_map
                if ordinal_adjective_map[expanded_context] == "True":
                    innings.append((inning_identifier_map[sent[i]], i))
    return innings


def sort_files_key(x):
    if "train" in x:
        file_index = int(x[5:7].strip("."))  # get the index of the train file
    else:
        file_index = -1  # valid and test
    return file_index


def filter_summaries(summary_entry, seen_output, test_seen_output):
    match_words = {"rain", "rains", "rained", "snow"}
    filter = False
    if len(summary_entry["summary"]) < 100:
        filter = True
    elif 100 < len(summary_entry["summary"]) < 300:
        if len(match_words.intersection(set(summary_entry["summary"]))) > 0:
            filter = True
    elif "_".join(summary_entry["summary"][:50]) in seen_output:  # retaining only one instance
        filter = True
    elif "_".join(summary_entry["summary"][:50]) in test_seen_output:  # retaining only one instance
        filter = True
    return filter


def extract_entities(entry, sent, all_ents, players=None, teams=None, cities=None, players_list=None, names_map=None):
    sent_ents = []
    sequential_entities = []
    matched_player_name = None
    team_name = None
    i = 0
    while i < len(sent):
        if sent[i] in all_ents:  # finds longest spans
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            candidate_entity = " ".join(sent[i:i + j - 1])
            if (candidate_entity in teams or candidate_entity in cities) and candidate_entity != "A":
                team_name = get_team_idx(entry, candidate_entity)
            elif candidate_entity in players:
                matched_player_name = get_player_idx(players_list, candidate_entity, names_map, sent[i - 1],
                                                     sent[i - 2])
            if matched_player_name is not None or team_name is not None:
                sent_ents.append((i, i + j - 1, candidate_entity))
                if matched_player_name is not None:
                    sequential_entities.append((matched_player_name, None))
                    matched_player_name = None
                elif team_name is not None:
                    sequential_entities.append(team_name)
                    team_name = None
            i += j - 1
        else:
            i += 1
    return sent_ents, sequential_entities


def chunks(input_list, chunk_size):
    for index in range(0, len(input_list), chunk_size):
        yield input_list[index: index + chunk_size]


def get_players_with_map(entry):
    player_team_map = {}
    bs = entry["box_score"]
    full_names = [x["full_name"] for x in bs]
    first_names = [x["first_name"] for x in bs]
    second_names = [x["last_name"] for x in bs]
    teams = [x["team"] for x in bs]
    players = []
    for k, _ in enumerate(full_names):
        players.append((full_names[k], second_names[k], first_names[k]))
        player_team_map[full_names[k]] = teams[k]
    return players, player_team_map


def replace_carmona(obj):
    def decode_dict(a_dict):
        for key, value in a_dict.items():
            try:
                if value == "Roberto Hernandez":
                    a_dict[key] = value.replace("Roberto Hernandez", "Fausto Carmona")
            except AttributeError:
                pass
        return a_dict
    return json.loads(json.dumps(obj), object_hook=decode_dict)
