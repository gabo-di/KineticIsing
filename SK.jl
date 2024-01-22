using PolyLog
using Integrals

function Gm(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * log(cosh(g + x*sqrtD))
end

function Gm0(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * abs(g + x*sqrtD)
end

function zGm(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * x * log(cosh(g + x*sqrtD))
end

function Gtanh(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * tanh(g + x*sqrtD)
end

function Gtanh0(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * sign(g + x*sqrtD)
end

function Gsign(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * sign(g + x*sqrtD)
end

function Gtanh2(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * (1 - tanh(g + x*sqrtD)^2)
end

function Gtanh20(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * (1 - sign(g + x*sqrtD)^2)
end

function Phi1(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * (-tanh(g + x*sqrtD)*(g + x*sqrtD) + log(2*cosh(g + x*sqrtD)))
end

function Phi(x, g, sqrtD)
    1/sqrt(2pi) * exp(-x^2/2) * -((g + x*sqrtD)*log(1 + exp(2*(g + x*sqrtD))) + 
    reli2(-exp(2*(g + x*sqrtD))))
end

function GaussianIntegral(F, g, sqrtD)
    if sqrtD==0
        return sqrt(2pi) * F(0, g, 0)
    else
        p = [g, sqrtD]
        # prob = IntegralProblem((x, p)->F(x, p[1], p[2]), -Inf, Inf, p) 
        prob = IntegralProblem((x, p)->F(x, p[1], p[2]), -5, 5, p) 
        sol = solve(prob, QuadGKJL(); reltol = 1e-5, abstol = 1e-5)
        return sol.u
    end
end

function Gmm(x, y, g, h1, h2, sqrtD, q)
    if 1-q^2>0
        return Gmm2(x, y, g, h1, h2, sqrtD, q)
    else
        return Gmm1(x, g, h1, h2, sqrtD)
    end
end

function Gmm2(x, y, g, h1, h2, sqrtD, q)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*(x*q + sqrt(1-q^2)*y)
    if a==b
        return 1/(2pi) * exp(-(x^2 + y^2)/2) * (g - tanh(g + a))
    else
        return 1/(2pi) * exp(-(x^2 + y^2)/2) * (g +
        ((exp(2b) + exp(2a))*(log(1+exp(2g + 2a)) - log(1+exp(2g + 2b))))/(exp(2b) - exp(2a)))
    end
end

function Gmm1(x, g, h1, h2, sqrtD)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*x
    if h1==h2
        return 1/sqrt(2pi) * exp(-x^2/2) * (g - tanh(g + a))
    else
        # return 1/sqrt(2pi) * exp(-x^2/2) * (g + 
        # (exp(2b) + exp(2a))^2 * (log(1+exp(2g + 2a)) - log(1+exp(2g + 2b)))/(exp(4b) + exp(4a)) )
        return 1/sqrt(2pi) * exp(-x^2/2) * (g + 
        (exp(2b) + exp(2a))*(log(1+exp(2g + 2a)) - log(1+exp(2g + 2b)))/(exp(2b) - exp(2a)))
        # ojo aqui tiene una expresion distinta que no causa indeterminacion 0/0
    end
end

function Gmm0(x, y, g, h1, h2, sqrtD, q)
    if 1-q^2>0
        return Gmm02(x, y, g, h1, h2, sqrtD, q)
    else
        return Gmm01(x, g, h1, h2, sqrtD)
    end
end

function Gmm02(x, y, g, h1, h2, sqrtD, q)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*(x*q + sqrt(1-q^2)*y)
    return 1/(2pi) * exp(-(x^2 + y^2)/2) * (g + sign(b-a)*(abs(g+a) - abs(g+b)))
end

function Gmm01(x, g, h1, h2, sqrtD)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*x
    return 1/sqrt(2pi) * exp(-x^2/2) * (g + sign(b-a)*(abs(g+a) - abs(g+b)))
end

function Gtanhtanh(x, y, g, h1, h2, sqrtD, q)
    if 1-q^2>0
        return Gtanhtanh2(x, y, g, h1, h2, sqrtD, q)
    else
        return Gtanhtanh1(x, g, h1, h2, sqrtD)
    end
end

function Gtanhtanh2(x, y, g, h1, h2, sqrtD, q)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*(x*q + sqrt(1-q^2)*y)
    return 1/(2pi) * exp(-(x^2 + y^2)/2) * (tanh(g+a) * tanh(g+b))
end

function Gtanhtanh1(x, g, h1, h2, sqrtD)
    a = h1 + sqrtD*x
    b = h2 + sqrtD*x
    return 1/sqrt(2pi) * exp(-x^2/2) * (tanh(g+a) * tanh(g+b))
end

function GaussianIntegral2D(F, g, h1, h2, sqrtD, q)
    if 1-q^2>0
        p = [g, h1, h2, sqrtD, q]
        # prob = IntegralProblem((x, p)->F(x[1], x[2], p[1], p[2], p[3], p[4], p[5]), [-Inf, -Inf], [Inf, Inf], p) 
        prob = IntegralProblem((x, p)->F(x[1], x[2], p[1], p[2], p[3], p[4], p[5]), [-5, -5], [5, 5], p) 
        sol = solve(prob, HCubatureJL(); reltol = 1e-5, abstol = 1e-5)
        return sol.u

    else
        p = [g, h1, h2, sqrtD, q]
        # prob = IntegralProblem((x, p)->F(x, 0, p[1], p[2], p[3], p[4], p[5]), -Inf, Inf, p) 
        prob = IntegralProblem((x, p)->F(x, 0, p[1], p[2], p[3], p[4], p[5]), -5, 5, p) 
        sol = solve(prob, QuadGKJL(); reltol = 1e-5, abstol = 1e-5)
        return sol.u
    end
end
