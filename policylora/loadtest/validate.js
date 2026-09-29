import http from "k6/http";
import { check } from "k6";

const tenants = ["base", "northline", "harbor", "cedar"];
const bodies = {
  base: "The Harbor Index Fund is a safe way to beat the market.",
  northline:
    "The Horizon Equity Fund returned 11% last year. Past performance does not guarantee future results.",
  harbor: "We could buy calls on the index this week.",
  cedar: "Brokerage cash is not FDIC insured.",
};

export const options = {
  scenarios: {
    mix: {
      executor: "constant-arrival-rate",
      rate: 20,
      timeUnit: "1s",
      duration: "30s",
      preAllocatedVUs: 20,
      maxVUs: 50,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
  },
};

export default function () {
  const tenant = tenants[__ITER % tenants.length];
  const payload = JSON.stringify({
    body: bodies[tenant],
    rulepack: "financial_services_client_communications",
    tenant_id: tenant,
    author_type: "ai",
  });
  const response = http.post(`${__ENV.BASE_URL || "http://127.0.0.1:8000"}/v1/validate`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(response, {
    "status 200": (res) => res.status === 200,
    "has verdict": (res) => res.json("verdict") !== "",
  });
}
