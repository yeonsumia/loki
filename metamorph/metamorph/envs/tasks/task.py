import os
from metamorph.config import cfg
from metamorph.envs.modules.agent import create_agent_xml
from metamorph.envs.tasks.locomotion import make_env_locomotion
from metamorph.envs.tasks.obstacle import make_env_obstacle
from metamorph.envs.tasks.push_box_incline import make_env_push_box_incline
from metamorph.envs.tasks.incline import make_env_incline
from metamorph.envs.tasks.bump import make_env_bump
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper
from metamorph.utils import file as fu


def make_env(agent_name, tmp_sample=False, init_xml=None):
    if init_xml is not None:
        xml = init_xml
    elif agent_name == "tmp":
        xml_path = os.path.join(
            cfg.ENV.WALKER_DIR, "{}.xml".format(agent_name)
        )
        xml = create_agent_xml(xml_path)
    elif tmp_sample:
        xml_path = os.path.join(
            cfg.ENV.WALKER_DIR, "tmp_samples", "xml", "{}.xml".format(agent_name)
        )
        xml = create_agent_xml(xml_path)
    elif cfg.LOKI.TRAIN:
        xml_path = os.path.join(
            cfg.ENV.WALKER_DIR, "xml_step", "0", "{}.xml".format(agent_name)
        )
        xml = create_agent_xml(xml_path)
    elif cfg.LOKI.EVAL:
        xml_path = os.path.join(
            cfg.ENV.WALKER_DIR, "xml_step", "final", "{}.xml".format(agent_name)
        )
        xml = create_agent_xml(xml_path)
    elif cfg.LOKI.FINETUNE:
        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml", "{}.xml".format(agent_name))
        xml = create_agent_xml(xml_path)
    env_func = "make_env_{}".format(cfg.ENV.TASK)
    env = globals()[env_func](xml, agent_name)

    # Add common wrappers in the end
    keys_to_keep = cfg.ENV.KEYS_TO_KEEP + cfg.MODEL.OBS_TYPES
    env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
    return env
