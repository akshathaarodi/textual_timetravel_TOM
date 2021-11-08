#!/usr/bin/env python3
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import actions
import copy
from enum import Enum
from .world import World
from .oracle import Oracle
from typing import List, Tuple
from . import actions
import numpy as np


def sample_question(
    oracle_start_state, oracle, agent1, agent2, obj, question, agent_order
):
    idx_dummy = [0]
    if question == "memory":
        action, trace = actions.MemoryAction(oracle_start_state, obj), "memory"
    elif question == "reality":
        action, trace = actions.RealityAction(oracle, obj), "reality"
    elif question == "belief":
        action = actions.BeliefSearchAction(oracle, agent1, agent2, obj)
        trace = f'second_order_{agent_order}_{"" if action.tom else "no_"}tom'
    elif question == "search":
        action = actions.SearchedAction(oracle, agent1, obj)
        trace = f'first_order_{agent_order}_{"" if action.tom else "no_"}tom'
    return action, trace

## Ak reverting changes
def sample_where_question(oracle, agent, obj, question):
    if question == "where_agent":
        action, trace = actions.WhereAgentAction(oracle, agent), "where_agent"
    elif question == "where_object":
        action, trace = actions.WhereObjectAction(oracle, obj), "where_object"
    return action, trace


class StoryType(Enum):
    true_belief = "true_belief"
    false_belief = "false_belief"
    second_order_false_belief = "second_order_false_belief"


def enter(oracle: Oracle, agent: str, observers: List[int], location: str):

    # if oracle.get_location(agent) == location:  # already in location
    #     return actions.LocationAction(oracle, (agent, location))
    # else:  # somewhere else, move this person into location
    #     return actions.EnterAction(oracle, (agent, location), observers)

    ## AK revertning changes first 3 lines keep comments
    #if oracle.get_location(agent) == location:  # already in location
    #    return actions.LocationAction(oracle, (agent, location))
    #else:  # somewhere else, move this person into location
    return actions.EnterAction(oracle, (agent, location), observers)


## AK reverting changes
def create_where_story(oracle, observers, obj, chapter, trace, where_stories, where_trace, where_agent_line):
    qtexts = []
    wtraces = []
    for observer in observers:
        qtext, qtrace = sample_where_question(oracle, observer, obj, "where_agent")
        qtexts += [qtext]
        wtraces += [qtrace]
    # if obj is not None:
    #     qtext, qtrace = sample_where_question(oracle, observer, obj, "where_object")
    #     qtexts += [qtext]
    #     wtraces += [qtrace]
    where_stories.append(chapter + qtexts)
    where_trace.append(trace + wtraces)
    where_agent_line.append(observers.copy())

def create_where_story_noise(oracle, agent, obj, where_story, where_trace, where_agent_line, location, alternative_loc):
    #Check if the agent is in the where agents. If yes, then do nothing, else, add a where question for it
    #loop through all the where stories first
    if not (oracle.get_location(agent) == location or oracle.get_location(agent) == alternative_loc):
        oracle.set_location(agent, None)
    qtext, qtrace = sample_where_question(oracle, agent, obj, "where_agent")
    if agent not in where_agent_line:
        where_story += [qtext]
        where_trace += [qtrace]
        where_agent_line += [agent]
    # for i in range(0,len(where_stories)):
    #     if (i>= from_idx and i<= to_idx) and agent not in where_agent_line[i]:
    #         qtext, qtrace = sample_where_question(oracle, agent, obj, "where_agent")
    #         where_stories[i] += [qtext]
    #         where_trace[i] += [qtrace]
    #         where_agent_line[i] += [agent]

def create_where_story_entry_noise(oracle, a3, obj, where_stories, where_trace, where_agent_line, noise_enter_idx, noise_exit_idx, entry_action, exit_action, enter_loc):
    qtext, qtrace = sample_where_question(oracle, a3, obj, "where_agent")
    oracle.set_location(a3, enter_loc)
    if noise_enter_idx == 0:
        where_story = [entry_action] + [qtext]
        where_stories.insert(noise_enter_idx, where_story)

        for i in range(1, len(where_stories)):
            where_story = where_stories[i]
            where_story.insert(noise_enter_idx, entry_action)
            where_story.append(qtext)
    else:
        where_story = where_stories[noise_enter_idx]
        where_story.insert(noise_enter_idx, entry_action)
        where_story.append(qtext)
        where_stories.insert(noise_enter_idx, where_story)

        for i in range(noise_enter_idx, len(where_stories)):
            where_story = where_stories[i]
            where_story.insert(noise_enter_idx, entry_action)
            where_story.append(qtext)

    oracle.set_location(a3, None)
    where_story = where_stories[noise_exit_idx]
    where_story.insert(noise_exit_idx, exit_action)
    where_story.append(qtext)
    where_stories.insert(noise_exit_idx, where_story)

    for i in range(noise_exit_idx, len(where_stories)):
        where_story = where_stories[noise_exit_idx]
        where_story.insert(noise_exit_idx, exit_action)
        where_story.append(qtext)

def get_obs_string(agents, agent_id):
    a = ''
    for x in set(agents):
        a += agent_id[x] + " "
    return a[:-1]

def generate_story(
    #world: World,
    ## AK revert
    world: World, opt,
) -> Tuple[List[List[actions.Action]], List[List[str]], StoryType]:
    oracle = Oracle(world)

    a1, a2, a3 = (world.get_agent() for _ in range(3))

    ## AK revert
    agent_id = {a1 : '1', a2 : '2', a3: '3' }

    story_type = StoryType.true_belief

    location = world.get_location()
    alternative_loc = world.get_location()

    # Get an initial object and container in the room
    obj = world.get_object()
    container_1 = world.get_container()
    container_2 = world.get_container()
    oracle.set_containers(location, [container_1, container_2])
    oracle.set_object_container(obj, container_1)

    trace = []
    chapter = []
    ## AK revert
    observer_list = []
    where_chapter = []
    where_trace = []
    where_stories = []
    where_agents = []
    where_agent_line = []
    oberservers_line = []
    # randomize the order in which agents enter the room
    first_agent = None
    agents = [(a1, 0), (a2, 1)]
    enter_observers = []
    np.random.shuffle(agents)
    agent_1, agent_2 = (x for _, x in agents)
    for agent, order in agents:
        chapter.append(enter(oracle, agent, enter_observers, location))
        ## AK revert
        observer_list.append(get_obs_string(enter_observers,agent_id))
        # Ak changed
        #enter_observers.append(agent)
        trace.append(f"enter_agent_{order}")
        ## AK revert
        where_agents.append(agent)
        create_where_story(oracle, where_agents, None, chapter, trace, where_stories, where_trace, where_agent_line)


    # announce location of object
    chapter.append(actions.ObjectLocAction(oracle, obj, [a for a, _ in agents]))
    ## AK revert
    observer_list.append(get_obs_string(enter_observers,agent_id))
    create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)
    start_state = copy.deepcopy(oracle)

    # Allow up to 2 location changes and 1 move.  Randomize the order...
    act_types = ["move"] + ["loc_change"] * np.random.randint(1, 3)
    np.random.shuffle(act_types)

    # If we move in the middle, this story moves into the false belief scenario.
    story_type = StoryType.false_belief if act_types[1] == "move" else story_type

    move_observers = {a1, a2}
    for i, act_type in enumerate(act_types):
        if act_type == "move":
            # move the object to container_2
            chapter.append(
                actions.MoveAction(oracle, (a1, obj, container_2), list(move_observers))
            )
            ## AK revert
            observer_list.append(get_obs_string(enter_observers,agent_id))
            trace.append(f"agent_0_moves_obj")
            ## AK revert
            create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)
        elif oracle.get_location(a2) == location:
            # a2 is in location, exit...
            chapter.append(actions.ExitedAction(oracle, a2))
            ## AK revert
            observer_list.append(get_obs_string(enter_observers,agent_id))
            move_observers.remove(a2)
            # Ak added
            if a2 in enter_observers:
                enter_observers.remove(a2)
            trace.append(f"agent_1_exits")
            ## AK revert
            create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)
        else:
            enter_observers = [a1]
            # Assuming this is the last action, then with 50% chance exit the moving actor
            if np.random.randint(0, 2) == 0 and i == len(act_types) - 1:
                story_type = (
                    StoryType.second_order_false_belief
                )  # this now is a second order falst belief
                # We can only do this if this is the last index of act_types, otherwise this agent
                # will try to move the object, but will be in the wrong location
                chapter.append(actions.ExitedAction(oracle, a1))
                ## AK revert
                observer_list.append(get_obs_string(enter_observers,agent_id))
                move_observers.remove(a1)
                enter_observers = []
                trace.append(f"agent_0_exits")
                ## AK revert
                create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)

            enter_loc = location if np.random.randint(0, 2) == 0 else alternative_loc
            # a2 already exited, re-enter same room, or a different one
            chapter.append(
                actions.EnterAction(oracle, (a2, enter_loc), enter_observers)
            )
            if enter_loc == location:
                move_observers.add(a2)
            else:
                enter_observers.remove(a2)
            ## AK revert
            observer_list.append(get_obs_string(move_observers,agent_id))
            trace.append(
                f"agent_1_reenters_" + ("alt_loc" if enter_loc != location else "loc")
            )
            ## AK revert
            create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)


    # generate indices for which person 3 should enter/exit
    indices = np.random.choice(
        np.arange(len(chapter) + 1), replace=False, size=np.random.randint(0, 3)
    )
    indices.sort()
    ## AK revert
    alter_loc =  False

    noise_enter_idx = 0
    noise_exit_idx = 0
    entered = False
    exited = False
    for idx, action in zip(indices, ["enter", "exit"]):
        if action == "exit":
            exit_action = actions.ExitedAction(oracle, a3)
            chapter.insert(idx, exit_action)
            ## AK revert
            if alter_loc:
                observer_list.insert(idx, agent_id[a3])
            else:
                observer_list.insert(idx,get_obs_string(enter_observers,agent_id))
            enter_observers.pop()  # remove person 3 from observers
            trace.insert(idx, f"agent_2_exits")
            where_trace.insert(idx, f"agent_2_enters")
            qtext, qtrace = sample_where_question(oracle, a3, obj, "where_agent")
            new_where_story = where_stories[idx-1].copy()
            new_where_story.insert(idx, exit_action)
            new_where_story[-1] = qtext
            where_stories.insert(idx, new_where_story)

            for i in range(idx+1, len(where_stories)):
                where_stories[i].insert(idx, exit_action)
                where_stories[i][-1] = qtext
            noise_exit_idx = idx
            exited = True
            new_where_agent_line = where_agent_line[idx - 1].copy()
            new_where_agent_line.append(a3)
            where_agent_line.insert(idx, new_where_agent_line)
            ## AK revert
            #create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)

        else:
            enter_loc = location if np.random.randint(0, 2) == 0 else alternative_loc
            entry_action = actions.EnterAction(oracle, (a3, enter_loc), enter_observers, False)
            chapter.insert(idx, entry_action)
            enter_observers.append(a3)
            ## AK revert
            if enter_loc == alternative_loc:
                observer_list.insert(idx,agent_id[a3])
                alter_loc = True
            else:
                observer_list.insert(idx,get_obs_string(enter_observers,agent_id))

            where_agents.append(a3)
            trace.insert(idx, f"agent_2_enters")
            noise_enter_idx = idx
            entered = True
            qtext, qtrace = sample_where_question(oracle, a3, obj, "where_agent")
            if idx == 0:
                new_where_story = [entry_action]
                new_where_agent_line = [a3]
            else:
                new_where_story = where_stories[idx-1].copy()
                new_where_story.insert(idx,entry_action)
                new_where_agent_line = where_agent_line[idx - 1].copy()
                new_where_agent_line.append(a3)
                #where_trace[idx-1].insert(idx, f"agent_2_enters")
            new_where_story.append(qtext)
            where_stories.insert(idx, new_where_story)

            where_agent_line.insert(idx, new_where_agent_line)

            for i in range(idx+1, len(where_stories)):
                where_stories[i].insert(idx,entry_action)
                where_stories[i].append(qtext)
                where_agent_line[i].append(a3)

            ## AK revert
            #create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)

    # if entered and exited:
    #     create_where_story_entry_noise(oracle, a3, obj, where_stories, where_trace, where_agent_line, noise_enter_idx, noise_exit_idx, entry_action, exit_action, enter_loc)

    # Add noise:
    indices = np.random.choice(
        np.arange(len(chapter) + 1), replace=False, size=np.random.randint(0, 3)
    )

    for idx in indices:
        person = np.random.choice([a1, a2, a3], 1)[0]
        things = world.get_all("objects")
        thing = np.random.choice(things, 1)[0]
        noise_action = actions.NoiseAction(oracle, person, thing)
        chapter.insert(idx, noise_action)
        ## AK revert
        observer_list.insert(idx, agent_id[person])
        #where_trace.insert(idx, f"agent_2_enters")
        adder = 0
        if oracle.get_location(person) != location or oracle.get_location(person) != alternative_loc or oracle.get_location(person) != None:
            oracle.set_location(person, None)
        if idx == 0:
            adder = 0
        else:
            adder = 1
            new_story = where_stories[idx-1].copy()
            new_story.insert(idx, noise_action)
            where_stories.insert(idx, new_story)
            new_where_agent_line = where_agent_line[idx - 1].copy()
            where_agent_line.insert(idx, new_where_agent_line)
        for i in range(idx+adder, len(where_stories)):
             where_story = where_stories[i]
             where_story.insert(idx, noise_action)
             if person not in where_agent_line[i]:
                 qtext, qtrace = sample_where_question(oracle, person, obj, "where_agent")
                 where_agent_line[i].append(person)
                 where_story.append(qtext)
        #     if i> idx:
        #     #if idx <= len(where_story):
        #         where_story.insert(idx, noise_action)
        #     where_stories[i] = where_story
                #create_where_story_noise(oracle, person, obj, where_story, where_trace[i], where_agent_line[i], location, alternative_loc)
                #create_where_story(oracle, where_agents, obj, chapter, trace, where_stories, where_trace, where_agent_line)
            #where_stories[i] = where_story


    stories, traces = [], []
    ## AK revert
    if opt.generate_where:
        stories += where_stories
        traces += where_trace

    for q in ["memory", "search", "belief", "reality"]:
        qtext, qtrace = sample_question(start_state, oracle, a1, a2, obj, q, agent_1)
        stories.append(chapter + [qtext])
        traces.append(trace + [qtrace])
        ## AK revert
        observer_list.append('')
    for q in ["search", "belief"]:
        qtext, qtrace = sample_question(start_state, oracle, a2, a1, obj, q, agent_2)
        stories.append(chapter + [qtext])
        traces.append(trace + [qtrace])
        ## AK revert
        observer_list.append('')
    ## AK revert
    #return stories, traces, story_type, observer_list
    return stories, traces, story_type, observer_list
    #return stories, traces, story_type
