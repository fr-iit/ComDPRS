import random

from Mechanism.Constraints import *

# ==================Mapping=================#

def sensitivity_Cp_fun(ep, k, m, y, index):
    L = LValue(ep, k, m, y, index)
    if (index == 1) or (index == 4):
        retVal = k * m * (2 * L - m)
        # print('retval: ',retVal)
        return retVal
    if (index == 2) or (index == 5):
        retVal = 2 * k * m * (2 * L - m) / np.pi
        return retVal
    if (index == 3) or (index == 6):
        retVal = k * m * (2 * L - m) / 2
        return retVal

def old_mapping_fromRealToL(input_value, sensitivity_f, lower, ep, k, m, y, index):
    L = LValue(ep, k, m, y, index)
    print('L : ', L)
    sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    print('sensitivity_Cp: ',sensitivity_Cp)
    C = sensitivity_Cp / sensitivity_f
    print('C :', C)
    mapped_value = (input_value - lower) * C - L
    # mapped_value_upper = (input_value - upper) * C + L
    print('input_value :', input_value)
    print('mapped_value: ',mapped_value)
    # print('mapped_value_upper: ', mapped_value_upper)

    return mapped_value

def mapping_fromRealToL(input_value, sensitivity_f, lower, L, sensitivity_Cp):
    # L = LValue(ep, k, m, y, index)
    # print('L : ', L)
    # sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    # print('sensitivity_Cp: ',sensitivity_Cp)
    C = sensitivity_Cp / sensitivity_f
    # print('C :', C)
    mapped_value = (input_value - lower) * C - L
    # mapped_value_upper = (input_value - upper) * C + L
    # print('input_value :', input_value)
    # print('mapped_value: ',mapped_value)
    # print('mapped_value_upper: ', mapped_value_upper)

    return mapped_value


def old_mapping_inverse_fromLToReal(input_value, sensitivity_f, lower, ep, k, m, y, index):
    L = LValue(ep, k, m, y, index)
    sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    C = sensitivity_Cp / sensitivity_f
    mapped_inverse_value = (input_value + L) / C + lower

    # print('mapped_inverse_value: ', mapped_inverse_value, " L: ", L, ' C: ', C, " sensitivity_Cp:", sensitivity_Cp, 'input: ', input_value)

    return mapped_inverse_value


def old_listmapping_inverse_fromLToReal(input_list, sensitivity_f, lower, upper, ep, k, m, y, index):
    mapped_inverse_list = []
    # i = 0
    # tmp = mapping_inverse_fromLToReal(input_list[i], sensitivity_f, lower, ep, k, m, y, index)
    # mapped_inverse_list.append(tmp)
    for i in range(len(input_list)):

        tmp = mapping_inverse_fromLToReal(input_list[i], sensitivity_f, lower, ep, k, m, y, index)
        # mapped_inverse_list.append(tmp)
        # additionally add this check to set the bound
        if tmp >= lower and tmp<= upper:
            mapped_inverse_list.append(tmp)

    random.shuffle(mapped_inverse_list)
    return mapped_inverse_list

def mapping_inverse_fromLToReal(input_value, sensitivity_f, lower, L, sensitivity_Cp):
    # L = LValue(ep, k, m, y, index)
    # sensitivity_Cp = sensitivity_Cp_fun(ep, k, m, y, index)
    C = sensitivity_Cp / sensitivity_f
    mapped_inverse_value = (input_value + L) / C + lower

    # print('mapped_inverse_value: ', mapped_inverse_value, " L: ", L, ' C: ', C, " sensitivity_Cp:", sensitivity_Cp, 'input: ', input_value)

    return mapped_inverse_value


def listmapping_inverse_fromLToReal(input_list, sensitivity_f, lower, upper, ep, k, m, y, index, L, Cp_Sen):
    mapped_inverse_list = []
    # i = 0
    # tmp = mapping_inverse_fromLToReal(input_list[i], sensitivity_f, lower, ep, k, m, y, index)
    # mapped_inverse_list.append(tmp)
    for i in range(len(input_list)):

        tmp = mapping_inverse_fromLToReal(input_list[i], sensitivity_f, lower, L, Cp_Sen)
        # mapped_inverse_list.append(tmp)
        # additionally add this check to set the bound
        if tmp >= lower and tmp<= upper:
            mapped_inverse_list.append(tmp)

    random.shuffle(mapped_inverse_list)
    return mapped_inverse_list


# ==========================================#