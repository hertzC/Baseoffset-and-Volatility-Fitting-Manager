'''
@Time: 2024/4/9 2:32 PM
@Author: Jincheng Gong
@Contact: Jincheng.Gong@hotmail.com
@File: wing_model_legacy.py
@Desc: Backward Compatibility Functions for Wing Model
'''

from typing import List, Dict
from .wing_model_parameters import WingModelParameters
from .wing_model import WingModel
from .wing_model_calibrator import WingModelCalibrator


def wing_model_durrleman_condition(vr_: float, sr_: float, pc_: float, cc_: float,
                                   dc_: float, uc_: float, dsm_: float, usm_: float,
                                   vcr_: float = 0, scr_: float = 0, ssr_: float = 100,
                                   atm_: float = 1, ref_: float = 1) -> List[List[float]]:
    """Backward compatibility function for Durrleman condition calculation"""
    params = WingModelParameters(vr_, sr_, pc_, cc_, dc_, uc_, dsm_, usm_, 
                                vcr_, scr_, ssr_, atm_, ref_)
    model = WingModel(params)
    moneyness, g_values = model.calculate_durrleman_condition()
    return [moneyness.tolist(), g_values.tolist()]


def wing_model(k: float, vr_: float, sr_: float, pc_: float, cc_: float,
               dc_: float, uc_: float, dsm_: float, usm_: float,
               vcr_: float = 0, scr_: float = 0, ssr_: float = 100,
               atm_: float = 1, ref_: float = 1) -> float:
    """Backward compatibility function for wing model volatility calculation"""
    params = WingModelParameters(vr_, sr_, pc_, cc_, dc_, uc_, dsm_, usm_, 
                                vcr_, scr_, ssr_, atm_, ref_)
    model = WingModel(params)
    return model.calculate_volatility(k)


def wing_model_calibrator(wing_model_params_list_input: List[float],
                          moneyness_inputs_list: List[float],
                          mkt_implied_vol_list: List[float],
                          mkt_vega_list: List[float],
                          is_bound_limit: bool = False,
                          epsilon: float = 1e-16) -> Dict:
    """Backward compatibility function for wing model calibration"""
    fixed_params = WingModelParameters(
        dc=wing_model_params_list_input[0],
        uc=wing_model_params_list_input[1],
        dsm=wing_model_params_list_input[2],
        usm=wing_model_params_list_input[3]
    )
    
    calibrator = WingModelCalibrator(
        enable_bounds=is_bound_limit,
        tolerance=epsilon
    )
    
    result = calibrator.calibrate(
        fixed_params=fixed_params,
        moneyness_list=moneyness_inputs_list,
        market_vol_list=mkt_implied_vol_list,
        market_vega_list=mkt_vega_list
    )
    
    result_dict = {
        "success": result.success,
        "vr_": result.parameters.vr,
        "sr_": result.parameters.sr,
        "pc_": result.parameters.pc,
        "cc_": result.parameters.cc
    }
    
    print(result_dict)
    return result_dict