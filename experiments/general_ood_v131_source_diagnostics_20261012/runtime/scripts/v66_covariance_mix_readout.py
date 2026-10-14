"""Normal-block validation of global/local conditional residual covariance."""
from collections import deque

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from sklearn.covariance import oas

from csbapr.belief.departure import NormalAgeReference, SETTINGS
from scripts import probe_general_ood_v43 as v43
from scripts.probe_general_ood_v42 import SCALE_FLOOR
from scripts.probe_general_ood_v56_reset_calibration import episode_ordinals
from scripts.v54_window_readout import COVARIANCE_RIDGE, WINDOW_STEPS, normalized_residuals, rolling_means
from scripts.v57_phase_readout import PhaseAgeReference, PhaseDepartureReadout
from scripts.v65_state_action_readout import NEIGHBORS, StateActionReference, OnlineStateActionReadout

BLOCK_STEPS = 1000
NAME = "state_action_mixed_covariance"


def normal_geometry(tree, moments, mean, scale, inputs):
    distances,indices = tree.query((inputs-mean)/scale,k=NEIGHBORS)
    values = moments[indices]
    covariances,shrinkages = [],[]
    dimension = moments.shape[1]
    for neighbors in values:
        covariance,weight = oas(neighbors)
        covariances.append(covariance+COVARIANCE_RIDGE*np.trace(covariance)/dimension*np.eye(dimension))
        shrinkages.append(weight)
    return values.mean(axis=1),np.asarray(covariances),dict(neighbor_radius=distances[:,-1],shrinkage=np.asarray(shrinkages))


def fit_windows(model,traces):
    values = []
    for trace in traces:
        ends = np.arange(WINDOW_STEPS-1,len(trace["dones"]),WINDOW_STEPS)
        values.append((rolling_means(model.inputs(trace))[ends],
                       rolling_means(normalized_residuals(model,trace))[ends],ends))
    return tuple(np.concatenate([v[i] for v in values]) for i in range(3))


def validation_samples(model,traces):
    lengths = [len(t["dones"]) for t in traces]
    assert len(set(lengths))==1 and lengths[0]%BLOCK_STEPS==0
    inputs,moments,ends = fit_windows(model,traces)
    starts = ends-WINDOW_STEPS+1
    offsets,covariances,folds = [],[],[]
    for a in range(0,lengths[0],BLOCK_STEPS):
        b = a+BLOCK_STEPS
        # Coupled branches share the holdout; purge every window touching its boundary.
        train = (ends<a)|(starts>=b)
        validation = (starts>=a)&(ends<b)
        assert train.sum()>=NEIGHBORS and validation.any()
        mean = inputs[train].mean(axis=0)
        scale = np.maximum(inputs[train].std(axis=0),SCALE_FLOOR)
        tree = cKDTree((inputs[train]-mean)/scale)
        centers,local,_ = normal_geometry(tree,moments[train],mean,scale,inputs[validation])
        offsets.append(moments[validation]-centers)
        covariances.append(local)
        folds.append(dict(heldout_transition_range=[a,b],training_windows=int(train.sum()),
            validation_windows=int(validation.sum()),purged_boundary_windows=int((~train&~validation).sum()),
            paired_branches_held_out_together=True))
    return np.concatenate(offsets),np.concatenate(covariances),folds


def normal_nll(weight,offsets,local,global_covariance):
    covariance = weight*global_covariance+(1-weight)*local
    signs,logdet = np.linalg.slogdet(covariance)
    assert np.all(signs>0), "non-positive-normal-mixture-covariance"
    solved = np.linalg.solve(covariance,offsets[...,None])[...,0]
    return float(.5*np.mean(logdet+np.einsum("ni,ni->n",offsets,solved)))


def select_weight(offsets,local,global_covariance):
    objective = lambda w:normal_nll(w,offsets,local,global_covariance)
    result = minimize_scalar(objective,bounds=(0.,1.),method="bounded")
    assert result.success, result.message
    candidates = [(0.,objective(0.)),(1.,objective(1.)),(float(result.x),float(result.fun))]
    weight,loss = min(candidates,key=lambda v:v[1])
    return weight,dict(selected_normal_nll=loss,local_only_normal_nll=candidates[0][1],
        global_only_normal_nll=candidates[1][1],bounded_solution=float(result.x),
        bounded_solution_normal_nll=float(result.fun),solver="scipy.optimize.minimize_scalar bounded plus endpoints",
        objective="mean Gaussian negative log likelihood excluding additive constant",
        optimizer_success=bool(result.success),optimizer_evaluations=int(result.nfev))


class MixedStateActionReference(StateActionReference):
    @classmethod
    def fit_normal(cls,model,original,fitting,calibration):
        reference = cls()
        reference.global_precision = original.global_precision.copy()
        reference.global_covariance = np.linalg.inv(reference.global_precision)
        reference.moments = original.moments.copy()
        reference.input_mean,reference.input_scale = original.input_mean.copy(),original.input_scale.copy()
        reference.tree = cKDTree(original.tree.data.copy())
        reference.fit_transitions = sum(len(t["dones"]) for t in fitting)
        inputs,moments,_ = fit_windows(model,fitting)
        np.testing.assert_allclose(moments,reference.moments,rtol=0,atol=1e-12)
        np.testing.assert_allclose((inputs-reference.input_mean)/reference.input_scale,reference.tree.data,rtol=0,atol=1e-12)
        offsets,local,folds = validation_samples(model,fitting)
        reference.global_weight,selection = select_weight(offsets,local,reference.global_covariance)
        reference.validation = dict(block_steps=BLOCK_STEPS,folds=folds,normal_validation_windows=len(offsets),**selection)
        values = [(rolling_means(normalized_residuals(model,t)),rolling_means(model.inputs(t)),
                   v43.episode_steps(t["dones"]),episode_ordinals(t["dones"])>0) for t in calibration]
        moments,inputs,ages,phases = (np.concatenate([v[i] for v in values]) for i in range(4))
        scores,diagnostic = reference.scores(moments,inputs)
        reference.score95 = {NAME:float(np.quantile(scores[NAME],SETTINGS["quantile"]))}
        assert reference.score95[NAME]>0
        signals = np.log1p(scores[NAME]/reference.score95[NAME])
        curves = [NormalAgeReference.fit_normal(signals[phases==p],ages[phases==p]) for p in (False,True)]
        assert all(c.ages[0]==0 and c.ages[-1]==999 for c in curves)
        reference.phases = {NAME:PhaseAgeReference(*curves)}
        reference.calibration_transitions = len(moments)
        reference.neighbor_radius95 = float(np.quantile(diagnostic["neighbor_radius"],SETTINGS["quantile"]))
        reference.calibration_geometry = dict(mean_shrinkage=float(diagnostic["shrinkage"].mean()),
            mean_covariance_trace=float(diagnostic["covariance_trace"].mean()))
        return reference

    def geometry(self,inputs):
        centers,local,diagnostic = normal_geometry(self.tree,self.moments,self.input_mean,self.input_scale,inputs)
        covariance = self.global_weight*self.global_covariance+(1-self.global_weight)*local
        diagnostic["covariance_trace"] = np.trace(covariance,axis1=1,axis2=2)
        return centers,np.linalg.inv(covariance),diagnostic

    def scores(self,moments,inputs):
        centers,precision,diagnostic = self.geometry(inputs)
        offsets = moments-centers
        return {NAME:np.einsum("ni,nij,nj->n",offsets,precision,offsets)/moments.shape[1]},diagnostic

    def metadata(self):
        metadata = super().metadata()
        metadata.update(covariance_estimator="normal-block-selected mixture of regularized global and local OAS covariance",
            covariance_formula="weight*inverse(v65_global_precision)+(1-weight)*v65_local_OAS_regularized_covariance",
            global_weight=self.global_weight,weight_selection=self.validation,
            calibration_used_for_weight_selection=False,ood_used_for_weight_selection=False)
        return metadata


class OnlineMixedStateActionReadout(OnlineStateActionReadout):
    def __init__(self,model,reference):
        self.model,self.reference,self.name = model,reference,NAME
        self.history,self.input_history = deque(maxlen=WINDOW_STEPS),deque(maxlen=WINDOW_STEPS)
        self.departure = PhaseDepartureReadout(reference.phases[NAME])
        self.signal = 0.
