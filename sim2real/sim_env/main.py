import os, time
import mujoco
import mujoco.viewer

print("RUNNING FILE =", __file__, flush=True)
print("CWD =", os.getcwd(), flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(HERE, "valve_test.xml")
print("XML =", xml_path, flush=True)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "valve_hinge")
dof = model.jnt_dofadr[jnt_id]
qadr = model.jnt_qposadr[jnt_id]


print("jnt_id =", jnt_id, "dof =", dof, "qadr =", qadr, flush=True)

print("before viewer", flush=True)


with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched", flush=True)
    step = 0
    while viewer.is_running():
        step += 1
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = 1.0

        mujoco.mj_forward(model, data)

        if step % 50 == 0:
            print("step", step,
                  "q", float(data.qpos[qadr]),
                  "qd", float(data.qvel[dof]),
                  "qacc", float(data.qacc[dof]),
                  flush=True)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
